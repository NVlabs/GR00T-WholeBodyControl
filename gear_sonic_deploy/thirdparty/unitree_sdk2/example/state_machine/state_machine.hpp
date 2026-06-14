#pragma once

#include <algorithm>

namespace unitree::common
{
    enum class STATES
    {
        DAMPING = 0,
        STAND = 1,
        CTRL = 2
    };

    class SimpleStateMachine
    {
    public:
        SimpleStateMachine(double pd_ratio_init = 0.1, double delta_pd = 0.005) : state(STATES::STAND), pd_ratio(pd_ratio_init), delta_pd(delta_pd) {}

        bool Stop()
        {
            state = STATES::DAMPING;
            pd_ratio = 0.0;
            return true;
        }

        bool Stand()
        {
            if (state == STATES::DAMPING || state == STATES::CTRL)
            {
                state = STATES::STAND;
                return true;
            }
            else
            {
                return false;
            }
        }

        bool Ctrl()
        {
            if (state == STATES::STAND && pd_ratio > 0.95)
            {
                state = STATES::CTRL;
                return true;
            }
            else
            {
                return false;
            }
        }

        void Standing(bool up = true)
        {
            if (state == STATES::STAND)
            {
                if (up)
                {
                    pd_ratio += delta_pd;
                }
                else
                {
                    pd_ratio -= delta_pd;
                }
                pd_ratio = std::max(0.0, std::min(1.0, pd_ratio));
            }
        }

        STATES state;
        double pd_ratio;
        double delta_pd;
    };
} // namespace unitree
