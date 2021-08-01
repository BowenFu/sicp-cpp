#include <cstdlib>
#include <optional>
#include <iostream>

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

template <typename IndexT, typename ValueT>
auto streamRef(IndexT i, Stream<ValueT> const& stream)
{
    assert(i >= 0);
    assert(stream.value());
    return i == 0 ? *stream.value() : streamRef(i - 1, stream.next());
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
    return (guess + x / guess) / T{2};
}

template <typename T>
auto sqrtStream(T x) -> Stream<T>
{
    Stream<T> result = {T{1}, 
        [=] {
            return streamMap([x](auto guess)
            {
                return sqrtImprove(guess, x);
            },
            sqrtStream(x));
        }};
    return result;
}


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
    printStream(sqrtStream<double>(2));
    return 0;
}