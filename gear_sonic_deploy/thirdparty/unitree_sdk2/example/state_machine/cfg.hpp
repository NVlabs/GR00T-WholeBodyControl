#include "unitree/common/json/jsonize.hpp"
#include <vector>
#include <iostream>

namespace unitree::common
{
    class ExampleCfg : public Jsonize
    {
    public:
        ExampleCfg() : kp(0), kd(0), dt(0)
        {
        }

        void fromJson(JsonMap &json)
        {
            FromJson(json["kp"], kp);
            FromJson(json["kd"], kd);
            FromJson(json["dt"], dt);
            FromAny<float>(json["init_pos"], init_pos);
        }

        void toJson(JsonMap &json) const
        {
            ToJson(kp, json["kp"]);
            ToJson(kd, json["kd"]);
            ToJson(dt, json["dt"]);
            ToAny<float>(init_pos, json["init_pos"]);
        }

        float kp;
        float kd;
        float dt;

        std::vector<float> init_pos;
    };
}