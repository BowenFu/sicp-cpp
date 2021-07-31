#include <cstdlib>
#include <optional>
#include <iostream>

template <typename ValueT>
struct Stream
{
    std::optional<ValueT> value;
    std::function<Stream<ValueT>()> next; // can wrap this as memo-func, + std::optional<ValueT> cached
};

template <typename T>
auto ones() -> Stream<T>
{
    return {1, ones<T>};
}

template <typename T>
auto addStreams(Stream<T> const& s1, Stream<T> const& s2) -> Stream<T>
{
    assert(static_cast<bool>(s1.value) == static_cast<bool>(s2.value));
    if (!s1.value)
    {
        return s1;
    }
    return {*s1.value + *s2.value, [=]{ return addStreams(s1.next(), s2.next()); }};
}

template <typename T>
auto naturals() -> Stream<T>
{
    return {0, [=]{ return addStreams(ones<T>(), naturals<T>()); }};
}

template <typename FuncT, typename ValueT>
auto streamMap(FuncT&& func, Stream<ValueT> const& stream)
{
    using RetT = std::invoke_result_t<FuncT, ValueT>;
    using ST = Stream<RetT>;
    if (!stream.value)
    {
        return ST{};
    }
    return ST{func(*stream.value), [=] { return streamMap(std::forward<FuncT>(func), stream.next()); }};
}

template <typename FuncT, typename ValueT>
auto streamForEach(FuncT&& func, Stream<ValueT> const& stream)
{
    static_assert(std::is_invocable_v<FuncT, ValueT>);
    if (!stream.value)
    {
        return;
    }
    func(*stream.value);
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
    if (!stream.value)
    {
        return ST{};
    }
    auto lambda = [=] { return streamFilter(std::forward<FuncT>(func), stream.next()); };
    if (func(*stream.value))
    {
        return ST{*stream.value, lambda};
    }
    return lambda();
}


int32_t main()
{
    // printStream(ones<int32_t>());
    auto const even = [](auto&& n) { return n%2 == 0;};
    printStream(streamFilter(even, naturals<int32_t>()));
    return 0;
}