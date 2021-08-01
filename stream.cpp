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
    Stream() = default;
    template <typename FuncT>
    Stream(ValueT value, FuncT next)
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
        std::cout << a << "+" << b << std::endl;
        return a + b; }, s1, s2);
}

template <typename T>
auto mulStreams(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    return streamMap([](T a, T b) { return a * b; }, s1, s2);
}

template <typename T>
auto ones() -> Stream<T>
{
    return {1, ones<T>};
}
template <typename T>
auto integers2() -> Stream<T>
{
    return {1, [=]{ return addStreams(ones<T>(), integers2<T>()); }};
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
constexpr auto scaleStream(Stream<T> const& stream, T factor)
{
    return streamMap([=](T v) { return v * factor; }, stream);
}

template <typename T>
constexpr auto double_() -> Stream<T>
{
    return Stream<T>{1, []{ return scaleStream(double_<T>(), T{2}); }};
}

template <typename T>
constexpr auto double2() -> Stream<T>
{
    return Stream<T>{T{1}, []{ return addStreams(double_<T>(), double_<T>()); }};
}

template <typename T>
constexpr auto factorials() -> Stream<T>
{
    return Stream<T>{T{1}, []{ return mulStreams(factorials<T>(), integers<T>().next()); }};
}

template <typename T>
constexpr auto partialSum() -> Stream<T>
{
    return Stream<T>{T{1}, []{ return addStreams(partialSum<T>(), integers<T>().next()); }};
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
constexpr auto s235() -> Stream<T>
{
    return {1, []
            { return merge(scaleStream(s235<T>(), T{2}), merge(scaleStream(s235<T>(), T{3}), scaleStream(s235<T>(), T{5}))); }};
}

int32_t main()
{
    // printStream(ones<int32_t>());
    // auto const even = [](auto&& n) { return n%2 == 0;};
    // printStream(streamFilter(even, streamEnumerateInterval(100000, 200000)));
    // printStream(fibs<int64_t>());
    // printStream(primes<int64_t>());
    printStream(fibs2<int64_t>());
    // sum(3, 8);
    // printStream(double_<int64_t>());
    // printStream(double2<int64_t>());
    // printStream(factorials<int64_t>());
    // printStream(partialSum<int64_t>());
    // printStream(s235<int64_t>());
    return 0;
}