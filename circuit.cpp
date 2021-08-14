#include <memory>
#include <cstdlib>
#include <functional>
#include <vector>
#include <string>
#include <iostream>
#include <queue>
#include <map>

class Wire
{
public:
    auto getSignal() const
    {
        return mValue;
    }
    void setSignal(bool newValue)
    {
        if (mValue == newValue)
        {
            return;
        }
        mValue = newValue;
        for (auto const& action : mActions)
        {
            action();
        }
    }
    void addAction(std::function<void()> func)
    {
        func();
        mActions.push_back(func);
    }
private:
    bool mValue{};
    std::vector<std::function<void()>> mActions;
};

using WirePtr = std::shared_ptr<Wire>;

auto makeWire()
{
    return std::make_shared<Wire>();
}

auto afterDelay(int32_t delay, std::function<void()> action) -> std::function<void()>;

void orGate(WirePtr in1, WirePtr in2, WirePtr out)
{
    auto const action = [out = std::move(out), in1, in2]{
        auto const newValue = in1->getSignal() || in2->getSignal();
        out->setSignal(newValue);
    };
    auto const orGateDelay = 5;
    auto const delayedAction = afterDelay(orGateDelay, action);
    in1->addAction(delayedAction);
    in2->addAction(delayedAction);
}

void andGate(WirePtr in1, WirePtr in2, WirePtr out)
{
    auto const action = [out = std::move(out), in1, in2]{
        auto const newValue = in1->getSignal() && in2->getSignal();
        out->setSignal(newValue);
    };
    auto const andGateDelay = 3;
    auto const delayedAction = afterDelay(andGateDelay, action);
    in1->addAction(delayedAction);
    in2->addAction(delayedAction);
}

void inverter(WirePtr in, WirePtr out)
{
    auto action = [in, out = std::move(out)]
    {
        auto newValue = !(in->getSignal());
        out->setSignal(newValue);
    };
    auto const inverterDelay = 2;
    in->addAction(afterDelay(inverterDelay, action));
}

auto halfAdder(WirePtr a, WirePtr b, WirePtr s, WirePtr c)
{
    auto d = makeWire();
    auto e = makeWire();
    orGate(a, b, d);
    andGate(a, b, c);
    inverter(c, e);
    andGate(d, e, s);
}

auto fullAdder(WirePtr a, WirePtr b, WirePtr cIn, WirePtr sum, WirePtr cOut)
{
    auto s = makeWire();
    auto c1 = makeWire();
    auto c2 = makeWire();
    halfAdder(b, cIn, s, c1);
    halfAdder(a, s, sum, c2);
    orGate(c1, c2, cOut);
}

auto adder(std::vector<WirePtr> A, std::vector<WirePtr> B, std::vector<WirePtr> S, WirePtr C)
{
    assert(A.size() == B.size());
    assert(A.size() == S.size());
    auto Cin = makeWire();
    for (size_t i = 0; i < A.size() - 1; ++i)
    {
        auto Cout = makeWire();
        fullAdder(A[i], B[i], Cin, S[i], Cout);
        Cin = Cout;
    }
    fullAdder(A.back(), B.back(), Cin, S.back(), C);
}

class Agenda
{
private:
    std::map<int32_t, std::queue<std::function<void()>>> mItems{};
    int32_t mTime{};
    Agenda() = default;
public:
    static Agenda& instance()
    {
        static Agenda theAgenda;
        return theAgenda;
    }
    bool empty() const
    {
        return mItems.empty();
    }
    auto pop()
    {
        auto& first = mItems.begin()->second;
        auto result = first.front();
        mTime = mItems.begin()->first;
        first.pop();
        if (first.empty())
        {
            mItems.erase(mItems.begin());
        }
        return result;
    }
    auto push(int32_t time, std::function<void()> action)
    {
        mItems[time].push(action);
        return action;
    }
    auto currentTime() const
    {
        return mTime;
    }
};

auto afterDelay(int32_t delay, std::function<void()> action) -> std::function<void()>
{
    return [=]
    {
        auto &theAgenda = Agenda::instance();
        theAgenda.push(theAgenda.currentTime() + delay, action);
    };
}

auto propagate()
{
    auto& theAgenda = Agenda::instance();
    if (theAgenda.empty())
    {
        return;
    }
    auto firstItem = theAgenda.pop();
    firstItem();
    propagate();
}

void probe(std::string const& name, WirePtr wire)
{
    auto const action = [=]
    {
        std::cout << name << " " 
        << Agenda::instance().currentTime()
        << " New-value = "
        << wire->getSignal() << std::endl;
    };
    wire->addAction(action);
}

int32_t main()
{
    auto input1 = makeWire();
    auto input2 = makeWire();
    auto sum = makeWire();
    auto carry = makeWire();
    // probe("input1", input1);
    // probe("input2", input2);
    probe("sum", sum);
    probe("carry", carry);
    //sum 0 New-value = 0
    // carry 0 New-value = 0

    halfAdder(input1, input2, sum, carry);
    input1->setSignal(1);
    propagate();
    // sum 5 New-value = 1
    input2->setSignal(1);
    propagate();
    // carry 11 New-value = 1
    // sum 16 New-value = 0

    //=== 2 bit adder
    auto a0 = makeWire();
    auto a1 = makeWire();
    auto b0 = makeWire();
    auto b1 = makeWire();
    auto s0 = makeWire();
    auto s1 = makeWire();
    auto c = makeWire();
    adder({a0, a1}, {b0, b1}, {s0, s1}, c);
    a0->setSignal(1);
    a1->setSignal(1);
    b1->setSignal(1);
    propagate();
    assert(s0->getSignal() == 1);
    assert(s1->getSignal() == 0);
    assert(c->getSignal() == 1);
    return 0;
}