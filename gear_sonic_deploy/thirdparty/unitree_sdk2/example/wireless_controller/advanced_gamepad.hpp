#pragma once

#include <stdint.h>
#include <cmath>
#include "unitree/idl/go2/WirelessController_.hpp"
// #include <go2_idl/WirelessController_.hpp>

namespace unitree::common
{
    // union for keys
    typedef union
    {
        struct
        {
            uint8_t R1 : 1;
            uint8_t L1 : 1;
            uint8_t start : 1;
            uint8_t select : 1;
            uint8_t R2 : 1;
            uint8_t L2 : 1;
            uint8_t F1 : 1;
            uint8_t F2 : 1;
            uint8_t A : 1;
            uint8_t B : 1;
            uint8_t X : 1;
            uint8_t Y : 1;
            uint8_t up : 1;
            uint8_t right : 1;
            uint8_t down : 1;
            uint8_t left : 1;
        } components;
        uint16_t value;
    } xKeySwitchUnion;

    // single button class
    class Button
    {
    public:
        Button() {}

        void update(bool state)
        {
            on_press = state ? state != pressed : false;
            on_release = state ? false : state != pressed;
            pressed = state;
        }

        bool pressed = false;
        bool on_press = false;
        bool on_release = false;
    };

    // full gamepad
    class Gamepad
    {
    public:
        Gamepad() {}

        void Update(unitree_go::msg::dds_::WirelessController_ &key_msg)
        {
            // update stick values with smooth and deadzone
            lx = lx * (1 - smooth) + (std::fabs(key_msg.lx()) < dead_zone ? 0.0 : key_msg.lx()) * smooth;
            rx = rx * (1 - smooth) + (std::fabs(key_msg.rx()) < dead_zone ? 0.0 : key_msg.rx()) * smooth;
            ry = ry * (1 - smooth) + (std::fabs(key_msg.ry()) < dead_zone ? 0.0 : key_msg.ry()) * smooth;
            ly = ly * (1 - smooth) + (std::fabs(key_msg.ly()) < dead_zone ? 0.0 : key_msg.ly()) * smooth;

            // update button states
            key.value = key_msg.keys();

            R1.update(key.components.R1);
            L1.update(key.components.L1);
            start.update(key.components.start);
            select.update(key.components.select);
            R2.update(key.components.R2);
            L2.update(key.components.L2);
            F1.update(key.components.F1);
            F2.update(key.components.F2);
            A.update(key.components.A);
            B.update(key.components.B);
            X.update(key.components.X);
            Y.update(key.components.Y);
            up.update(key.components.up);
            right.update(key.components.right);
            down.update(key.components.down);
            left.update(key.components.left);
        }

        float smooth = 0.03;
        float dead_zone = 0.01;

        float lx = 0.;
        float rx = 0.;
        float ry = 0.;
        float ly = 0.;

        Button R1;
        Button L1;
        Button start;
        Button select;
        Button R2;
        Button L2;
        Button F1;
        Button F2;
        Button A;
        Button B;
        Button X;
        Button Y;
        Button up;
        Button right;
        Button down;
        Button left;

    private:
        xKeySwitchUnion key;
    };
} // namespace unitree::common