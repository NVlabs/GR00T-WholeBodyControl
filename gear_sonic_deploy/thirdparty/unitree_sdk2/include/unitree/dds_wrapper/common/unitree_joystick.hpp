// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <math.h>
#include <stdint.h>
#include <iostream>

namespace unitree
{
namespace common
{
    
// ***************************** Unitree Joystick Data Type ***************************** //
typedef union
{
  struct
  {
    uint8_t R1 : 1;
    uint8_t L1 : 1;
    uint8_t Start : 1;
    uint8_t Select : 1;
    uint8_t R2 : 1;
    uint8_t L2 : 1;
    uint8_t f1 : 1;
    uint8_t f2 : 1;
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
} BtnUnion;

typedef struct
{
  uint8_t head[2];
  BtnUnion btn;
  float lx;
  float rx;
  float ry;
  float L2;
  float ly;
} BtnDataStruct;         

typedef union
{
  BtnDataStruct RF_RX;
  uint8_t buff[40];
}REMOTE_DATA_RX;

// ***************************** Button & Axis ***************************** //

template <typename T> // int, bool, string
class Button
{
public:
  void operator()(const T& data){ // update
    pressed = (data != dataNull_);
    on_pressed = (pressed && data_ == dataNull_);
    on_released = (!pressed && data_ != dataNull_);
    data_ = data;
  }
  const T& operator()() { return data_; }

  bool pressed = false;
  bool on_pressed = false;
  bool on_released = false;
private:
  T data_{}, dataNull_{};
};

class Axis
{
public:
  void operator()(const float& data){ // update
    auto data_deadzone = std::fabs(data) < deadzone ? 0.0 : data;
    double new_data = data_ * (1.0 - smooth) + data_deadzone * smooth;
    pressed = (new_data > threshold);
    on_pressed = (pressed && data_ < threshold);
    on_released = (!pressed && data_ > threshold);
    data_ = new_data;
  }

  const float& operator()() { return data_; }

  float smooth = 0.03;
  float deadzone = 0.01;

  // Change an axis value to a button
  bool pressed = false;
  bool on_pressed = false;
  bool on_released = false;
  float threshold{0.5};
private:
  float data_{};
};

// ***************************** Unitree Joystick Interface ***************************** //

class UnitreeJoystick
{
public:
  UnitreeJoystick() = default;
  
  // Adopts standard joystick key names
  Button<int> back;
  Button<int> start;
  Button<int> LS;
  Button<int> RS;
  Button<int> LB;
  Button<int> RB;
  Button<int> A;
  Button<int> B;
  Button<int> X;
  Button<int> Y;
  Button<int> up;
  Button<int> down;
  Button<int> left;
  Button<int> right;
  Button<int> F1;
  Button<int> F2;
  Axis lx;
  Axis ly;
  Axis rx;
  Axis ry;
  Axis LT;
  Axis RT;

  virtual void update(){};

  void extract(const REMOTE_DATA_RX& key)
  {
    back(key.RF_RX.btn.components.Select);
    start(key.RF_RX.btn.components.Start);
    LB(key.RF_RX.btn.components.L1);
    RB(key.RF_RX.btn.components.R1);
    F1(key.RF_RX.btn.components.f1);
    F2(key.RF_RX.btn.components.f2);
    A(key.RF_RX.btn.components.A);
    B(key.RF_RX.btn.components.B);
    X(key.RF_RX.btn.components.X);
    Y(key.RF_RX.btn.components.Y);
    up(key.RF_RX.btn.components.up);
    down(key.RF_RX.btn.components.down);
    left(key.RF_RX.btn.components.left);
    right(key.RF_RX.btn.components.right);
    LT(key.RF_RX.btn.components.L2);
    RT(key.RF_RX.btn.components.R2);
    lx(key.RF_RX.lx);
    ly(key.RF_RX.ly);
    rx(key.RF_RX.rx);
    ry(key.RF_RX.ry);
  }

  REMOTE_DATA_RX combine()
  {
    REMOTE_DATA_RX key;
    key.RF_RX.btn.components.Select = back();
    key.RF_RX.btn.components.Start = start();
    key.RF_RX.btn.components.L1 = LB();
    key.RF_RX.btn.components.R1 = RB();
    key.RF_RX.btn.components.f1 = F1();
    key.RF_RX.btn.components.f2 = F2();
    key.RF_RX.btn.components.A = A();
    key.RF_RX.btn.components.B = B();
    key.RF_RX.btn.components.X = X();
    key.RF_RX.btn.components.Y = Y();
    key.RF_RX.btn.components.up = up();
    key.RF_RX.btn.components.down = down();
    key.RF_RX.btn.components.left = left();
    key.RF_RX.btn.components.right = right();
    key.RF_RX.btn.components.L2 = LT() > 0.5f ? 1 : 0;
    key.RF_RX.btn.components.R2 = RT() > 0.5f ? 1 : 0;
    key.RF_RX.lx = lx();
    key.RF_RX.ly = ly();
    key.RF_RX.rx = rx();
    key.RF_RX.ry = ry();
    return key;
  }
};

} // namespace common
} // namespace unitree