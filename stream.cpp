#include <cstdlib>
#include <optional>
#include <iostream>
#include <numeric>

// This is a must-have for recursive definition of stream.
template <typename ValueT>
class MemoFunction;

template <typename ValueT>
class Stream
{
    std::optional<ValueT> mValue;
    std::shared_ptr<MemoFunction<ValueT>> mNext;
public:
    using ValueType = ValueT;
    constexpr Stream() = default;
    template <typename FuncT>
    constexpr Stream(ValueT value, FuncT next)
        : mValue{value}, mNext{std::make_shared<MemoFunction<ValueT> >(next)}
    {}
    auto value() const
    {
        return mValue;
    }
    auto next() const
    {
        return (*mNext)();
    }
};

template <typename ValueT>
class MemoFunction
{
    mutable std::mutex mMutex;
    mutable std::optional<Stream<ValueT>> mCache;
    std::function<Stream<ValueT>()> mFunc;
public:
    MemoFunction() = default;
    template <typename FuncT>
    MemoFunction(FuncT func)
    : mFunc{std::move(func)}
    {}
    auto operator()() const
    {
        std::lock_guard<std::mutex> l{mMutex};
        auto cache = mCache;
        if (cache)
        {
            return *cache;
        }
        auto result = mFunc();
        mCache = result;
        return result;
    }
};

template <typename FuncT, typename ValueT, typename... StreamTs>
auto streamMap(FuncT&& func, Stream<ValueT> const& stream, StreamTs const&... rest)
{
    using RetT = std::invoke_result_t<FuncT, ValueT, typename StreamTs::ValueType...>;
    using ST = Stream<RetT>;
    auto hasValue = stream.value().has_value();
    (assert(rest.value().has_value() == hasValue), ...);
    if (!hasValue)
    {
        return ST{};
    }
    return ST{func(*stream.value(), (*rest.value())...), [=] {
        return streamMap(func, stream.next(), rest.next()...); }};
}

template <typename FuncT, typename ValueT>
auto streamForEach(FuncT&& func, Stream<ValueT> const& stream)
{
    static_assert(std::is_invocable_v<FuncT, ValueT>);
    if (!stream.value())
    {
        return;
    }
    func(*stream.value());
    streamForEach(std::forward<FuncT>(func), stream.next());
}

template <typename T>
std::ostream& operator<<(std::ostream &o, std::pair<T, T> const& p)
{
    return o << p.first << ", " << p.second;
}

template <typename ValueT>
auto printStream(Stream<ValueT> const& stream)
{
    streamForEach([](auto &&value)
                  { std::cout << value << std::endl; }, stream);
}

template <typename FuncT, typename ValueT>
auto streamFilter(FuncT&& func, Stream<ValueT> const& stream)
{
    using RetT = std::invoke_result_t<FuncT, ValueT>;
    static_assert(std::is_convertible_v<RetT, bool>);
    using ST = Stream<ValueT>;
    if (!stream.value())
    {
        return ST{};
    }
    auto lambda = [=] { return streamFilter(std::forward<FuncT>(func), stream.next()); };
    if (func(*stream.value()))
    {
        return ST{*stream.value(), lambda};
    }
    return lambda();
}

template <typename ValueT, typename IndexT>
auto streamRef(Stream<ValueT> const& stream, IndexT i) -> ValueT
{
    assert(i >= 0);
    assert(stream.value());
    return i == 0 ? *stream.value() : streamRef(stream.next(), i - 1);
}

template <typename T>
auto streamEnumerateInterval(T start, T end)
{
    if (start == end)
    {
        return Stream<T>{};
    }
    return Stream<T>{start, [=]{ return streamEnumerateInterval(start + 1, end); }};
}

template <typename T>
auto integersStartingFrom(T start) -> Stream<T>
{
    return {start, [=]{ return integersStartingFrom(start + 1); }};
}

template <typename T>
auto integers()
{
    return integersStartingFrom(T{1});
}

template <typename T>
constexpr auto fibgen(T a, T b) -> Stream<T>
{
    return {a, [=]{ return fibgen(b, a + b); }};
}

template <typename T>
constexpr auto fibs()
{
    return fibgen<T>(0, 1);
}

template <typename T>
constexpr auto sieve(Stream<T> const& stream) -> Stream<T>
{
    auto const& value = *stream.value();
    auto const& pred = [=](auto&& v) { return v%value != 0; };
    return Stream<T>{value, [=]
                     { return sieve(streamFilter(pred, stream.next())); }};
}

template <typename T>
constexpr auto primes()
{
    return sieve<T>(integersStartingFrom(T{2}));
}

template <typename T>
auto addStreams(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    return streamMap([](T a, T b) {
        // std::cout << a << "+" << b << std::endl;
        return a + b; }, s1, s2);
}

template <typename T>
auto subtractStreams(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    return streamMap([](T a, T b) {
        // std::cout << a << "-" << b << std::endl;
        return a - b; }, s1, s2);
}

template <typename T>
auto mulStreams(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    return streamMap([](T a, T b) {
        // std::cout << a << "*" << b << std::endl;
        return a * b; }, s1, s2);
}

template <typename T>
constexpr auto scaleStream(Stream<T> const& stream, T factor)
{
    return streamMap([=](T v) {
        // std::cout << v << "*" << factor << std::endl;
        return v * factor;
        }, stream);
}

template <typename T>
auto partialSumStream(Stream<T> const& stream) -> Stream<T>
{
    if (!stream.value())
    {
        return {};
    }
    Stream<T> result = {*stream.value(), [=]{ return addStreams(partialSumStream<T>(stream), stream.next()); }};
    return result;
}

template <typename T>
auto ones() -> Stream<T>
{
    return {1, ones<T>};
}
template <typename T>
auto integers2() -> Stream<T>
{
    // static variable for memorizing.
    static Stream<T> result = {T{1}, [=]{ return addStreams(ones<T>(), integers2<T>()); }};
    return result;
}

template <typename T>
auto fibs2() -> Stream<T>
{
    static Stream<T> result = {T{0}, [=]
            { return Stream<T>{T{1}, [=]
                               { return addStreams(fibs2<T>(), fibs2<T>().next()); }}; }};
    return result;
}

template <typename T>
auto double_() -> Stream<T>
{
    static Stream<T> result = {T{1}, []{ return scaleStream(double_<T>(), T{2}); }};
    return result;
}

template <typename T>
auto double2() -> Stream<T>
{
    static Stream<T> result = Stream<T>{T{1}, []{ return addStreams(double2<T>(), double2<T>()); }};
    return result;
}

template <typename T>
auto factorials() -> Stream<T>
{
    static Stream<T> result = {T{1}, []{ return mulStreams(factorials<T>(), integers<T>().next()); }};
    return result;
}

template <typename T>
auto partialSum() -> Stream<T>
{
    static Stream<T> result = {T{1}, []{ return addStreams(partialSum<T>(), integers<T>().next()); }};
    return result;
}

template <typename T>
constexpr auto merge(Stream<T> const& s1, Stream<T> const& s2)
{
    if (!s1.value())
    {
        return s2;
    }
    if (!s2.value())
    {
        return s1;
    }
    if (*s1.value() < *s2.value())
    {
        return Stream<T>{*s1.value(), [=]{ return merge(s1.next(), s2); }};
    }
    if (*s2.value() > *s1.value())
    {
        return Stream<T>{*s2.value(), [=]{ return merge(s2.next(), s1); }};
    }
    return Stream<T>{*s1.value(), [=]{ return merge(s1.next(), s2.next()); }};
}

template <typename T>
auto s235() -> Stream<T>
{
    static Stream<T> result = {T{1}, []
            { return merge(scaleStream(s235<T>(), T{2}), merge(scaleStream(s235<T>(), T{3}), scaleStream(s235<T>(), T{5}))); }};
    return result;
}

template <typename T>
auto integrateSeries(Stream<T> const& stream) -> Stream<T>
{
    return streamMap(
        [](T a, T b) { return a / b;},
        stream,
        integers<T>()
    );
}

template <typename T>
auto expSeries() -> Stream<T>
{
    static Stream<T> result = {T{1}, []
            { return integrateSeries(expSeries<T>()); }};
    return result;
}

template <typename T>
auto cosSeries() -> Stream<T>;

template <typename T>
auto sinSeries() -> Stream<T>
{
    static Stream<T> result = {T{0}, []
            { return integrateSeries(cosSeries<T>()); }};
    return result;
}

template <typename T>
auto cosSeries() -> Stream<T>
{
    static Stream<T> result = {T{1}, []
            { return scaleStream(integrateSeries(sinSeries<T>()), T{-1}); }};
    return result;
}

template <typename T>
auto mulSeries(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    assert(s1.value());
    assert(s2.value());
    auto s1v = *s1.value();
    auto s2v = *s2.value();
    Stream<T> result = {s1v * s2v, // constant
                        [=]
                        {
                            return addStreams(
                                scaleStream(s2.next(), s1v),
                                mulSeries(s2, s1.next()));
                        }};
    return result;
}

// S * X = 1
// (a0 + SR) * X = 1
// a0 * X + SR * X = 1
// a0 * X = 1 - SR * X
// X = (1 - SR * X) / a0
template <typename T>
auto invertSeries(Stream<T> const& stream) -> Stream<T>
{
    T a0 = *stream.value();
    return Stream<T>{a0, [=] { return scaleStream(mulSeries(stream.next(), invertSeries(stream)), -a0); }};
}

template <typename T>
auto divSeries(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    Stream<T> result = mulSeries(s1, invertSeries(s2));
    return result;
}

template <typename T>
auto tanSeries() -> Stream<T>
{
    static Stream<T> result = divSeries(sinSeries<T>(), cosSeries<T>());
    return result;
}

template <typename T>
auto sqrtImprove(T guess, T x)
{
    // std::cout << "sqrtImprove: " << guess << std::endl;
    return (guess + x / guess) / T{2};
}

template <typename T>
auto sqrtStream(T x) -> Stream<T>
{
    Stream<T> result = {T{1}, 
        [&result, x] // &result memory issue? used shared_ptr<Stream<T>> instead
        {
            return streamMap([x](auto guess)
            {
                return sqrtImprove(guess, x);
            },
            result);
        }};
    return result;
}

template <typename T>
auto piSummands(int32_t n) -> Stream<T>
{
    return {T{1} / n, [=]{
        return scaleStream(piSummands<T>(n + 2), T{-1});
    }};
}

template <typename T>
auto piStream() -> Stream<T>
{
    static Stream<T> result = partialSumStream(scaleStream(piSummands<T>(T{1}), T{4}));
    return result;
}

template <typename T>
auto eulerTransform(Stream<T> const& stream) -> Stream<T>
{
    auto s0 = streamRef(stream, 0); // Sn-1
    auto s1 = streamRef(stream, 1); // Sn
    auto s2 = streamRef(stream, 2); // Sn+1
    return {
        s2 - (s2 - s1)*(s2 - s1) / (s0 - 2*s1 + s2),
        [=]{
            return eulerTransform(stream.next());
        }
    };
}

template <typename TransformT, typename T>
auto makeTableau(TransformT transform, Stream<T> const& stream) -> Stream<Stream<T>>
{
    return {stream, [=]{ return makeTableau(transform, transform(stream)); }};
}

template <typename T, typename TransformT = std::function<Stream<T>(Stream<T> const&)>>
auto acceleratedSequence(TransformT transform, Stream<T> const& stream) -> Stream<T>
{
    return streamMap([](auto s) { return *s.value(); }, makeTableau(transform, stream));
}

template <typename T>
auto iterate(T last, Stream<T> const& stream, T tol) -> T
{
    assert(stream.value());
    auto v = *stream.value();
    auto diff = v - last;
    if (diff < tol && diff > -tol)
    {
        return v;
    }
    return iterate(v, stream.next(), tol);
}

template <typename T>
auto streamLimit(Stream<T> const& stream, T tol)
{
    assert(stream.value());
    auto v = *stream.value();
    return iterate(v, stream.next(), tol);
}

template <typename T>
auto pi(T tol)
{
    return streamLimit(acceleratedSequence(eulerTransform<T>, piStream<T>()), tol);
}

template <typename T>
auto sqrt(T x, T tol)
{
    return streamLimit(sqrtStream<T>(x), tol);
}

template <typename T>
auto interleave(Stream<T> const& s, Stream<T> const& t) -> Stream<T>
{
    if (!s.value())
    {
        return t;
    }
    return {*s.value(), [=]{
        return interleave(t, s.next());
    }};
}

template <typename T>
auto pairs(Stream<T> const& s, Stream<T> const& t) -> Stream<std::pair<T, T>>
{
    return {{*s.value(), *t.value()},
            [=]
            {
                return interleave(
                    streamMap([=](auto x)
                              { return std::pair<T, T>{*s.value(), x}; },
                              t.next()),
                    pairs(s.next(), t.next()));
            }};
}

//===========

template <typename T, typename FuncT = std::function<Stream<T>()>>
auto integral(FuncT const& delayedIntegrand, T initialValue, T dt)
{
    auto integ = std::make_shared<Stream<T>>();
    *integ = Stream<T>{ 
        initialValue,
        [=]{
            return addStreams(scaleStream(delayedIntegrand(), dt), *integ);
        }
     };
    return *integ;
}

template <typename T>
auto vOfRC(T R, T C, T dt)
{
    return [=](Stream<T> const &i, T v0)
    {
        return addStreams(
            scaleStream(i, R),
            integral([=]{ return scaleStream(i, T{1} / C); }, v0, dt));
    };
}

template <typename T, typename FuncT = std::function<T(T)>>
auto solve(FuncT f, T y0, T dt)
{
    auto dy = [=] (auto y)
    {
        return streamMap(f, y);
    };
    auto y = std::make_shared<Stream<T>>();
    *y = integral([=]{ return dy(*y); }, y0, dt);
    return *y;
}

template <typename T>
auto RLC(T R, T L, T C, T dt)
{
    return [=](T Vc0, T Il0)
    {
        auto dVc = [=] (auto Il)
        {
            return scaleStream(Il, -T{1}/C);
        };
        auto dIl = [=] (auto Vc, auto Il)
        {
            return addStreams(scaleStream(Vc, T{1}/L), scaleStream(Il, -R/L));
        };
        auto Vc = std::make_shared<Stream<T>>();
        auto Il = std::make_shared<Stream<T>>();
        *Vc = integral([=]{ return dVc(*Il); }, Vc0, dt);
        *Il = integral([=]{ return dIl(*Vc, *Il); }, Il0, dt);
        return std::make_pair(*Vc, *Il);
    };
}

template <typename T>
auto randomNumbers(T seed)
{
    auto randUpdate = [](T v)
    {
        return ((v * T{1103515245} + T{12345}) / T{6533}) % T{32174682};
    };

    auto result = std::make_shared<Stream<T>>();
    *result = Stream<T>{seed, [=]{ return streamMap(randUpdate, *result); }};
    return *result;
}

template <typename FuncT, typename ValueT>
auto mapSuccessivePairs(FuncT&& func, Stream<ValueT> const& stream) -> Stream<std::invoke_result_t<FuncT, ValueT, ValueT>>
{
    auto v1 = *stream.value();
    auto next = stream.next();
    auto v2 = *next.value();
    auto v = func(v1, v2);
    return {
        v,
        [=] { return mapSuccessivePairs(func, next.next()); }
    };
}

template <typename T>
auto cesaroStream()
{
    return mapSuccessivePairs( 
        [](T r1, T r2) -> bool { return std::gcd(r1, r2) == 1; },
        randomNumbers(T{11}) // random seed
     );
}

auto monteCarlo(Stream<bool> const& experimentStream, int32_t passed, int32_t failed) -> Stream<double>
{
    auto next = [=](int32_t passed, int32_t failed)
    {
        return Stream<double>{
            double(passed) / (passed + failed),
            [=] {
                return monteCarlo(experimentStream.next(), passed, failed);
            }
        };
    };
    if (*experimentStream.value())
    {
        return next(passed+1, failed);
    }
    return next(passed, failed+1);
}

auto pi_ = streamMap(
    [](auto p)
    {
        return sqrt(6.0/p, 1e-6);
    },
    monteCarlo(cesaroStream<int64_t>(), int64_t{0}, int64_t{0})
);

int32_t main()
{
    // auto const even = [](auto&& n) { return n%2 == 0;};
    // printStream(streamFilter(even, streamEnumerateInterval(100000, 200000)));
    // printStream(fibs<int64_t>());
    // printStream(primes<int64_t>());
    // printStream(integers<int32_t>());
    // printStream(fibs2<int64_t>());
    // printStream(double_<int64_t>());
    // printStream(double2<int64_t>());
    // printStream(factorials<int64_t>());
    // printStream(partialSum<int64_t>());
    // printStream(partialSumStream(ones<int64_t>()));
    // printStream(s235<int64_t>());
    // printStream(integers2<int64_t>());
    // printStream(expSeries<double>());
    // printStream(sinSeries<double>());
    // printStream(cosSeries<double>());
    // printStream(mulSeries(cosSeries<double>(), cosSeries<double>()));
    // printStream(addStreams(mulSeries(sinSeries<double>(), sinSeries<double>()), mulSeries(cosSeries<double>(), cosSeries<double>())));
    // printStream(invertSeries(cosSeries<double>()));
    // printStream(tanSeries<double>());
    // printStream(sqrtStream<double>(2));
    // printStream(acceleratedSequence(eulerTransform<long double>, piStream<long double>()));
    // std::cout << pi(1e-10) << std::endl;
    // std::cout << sqrt(2.0, 1e-10) << std::endl;
    // printStream(pairs(integers<int32_t>(), integers<int32_t>()));
    // auto RC1 = vOfRC(5.0, 1.0, 0.1);
    // printStream(RC1(ones<double>(), 5));
    // std::cout << streamRef(solve([](auto y) { return y; }, 1.0, 0.001), 1000) << std::endl;
    // auto RLC1 = RLC(1.0, 1.0, 0.2, 0.1);
    // auto RLC1_ = RLC1(10.0, 0.0);
    /*
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.loadtxt("log")
    plt.plot(x[:1000])
    plt.savefig("x.png")
    */
    // printStream(RLC1_.first);
    // printStream(RLC1_.second);
    // printStream(cesaroStream<int32_t>());
    printStream(pi_);
    return 0;
}