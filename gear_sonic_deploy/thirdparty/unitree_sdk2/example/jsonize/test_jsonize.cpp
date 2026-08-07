#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace common
{
class Test : public Jsonize
{
public:
    Test() : x(0), y(0), z(0)
    {}

    void fromJson(JsonMap& json)
    {
        FromJson(json["x"], x);
        FromJson(json["y"], y);
        FromJson(json["z"], z);
    }

    void toJson(JsonMap& json) const
    {
        ToJson(x, json["x"]);
        ToJson(y, json["y"]);
        ToJson(z, json["z"]);
    }

public:
    int x;
    int y;
    int z;
};
}
}

using namespace unitree::common;

int main()
{
    std::vector<Test> vec;

    Test t1;
    t1.x = 1;
    t1.y = 2;
    t1.z = 3;

    vec.push_back(t1);

    Test t2;
    t2.x = 10;
    t2.y = 20;
    t2.z = 30;

    vec.push_back(t2);

    Test t3;
    t3.x = 100;
    t3.y = 200;
    t3.z = 300;

    vec.push_back(t3);

    std::string s = ToJsonString(vec);

    std::cout << "s=" << s << std::endl;

    std::vector<Test> vec2;
    FromJsonString(s, vec2);

    size_t i, size = vec2.size();
    for (i=0; i<size; i++)
    {
        const Test& t = vec2[i];
        std::cout << "t.x=" << t.x << ", t.y=" << t.y << ", t.z=" << t.z << std::endl;
    }

    std::map<std::string,std::string> tmp;

    tmp["abc"] = "a1b1c1";
    tmp["xyz"] = "x1y1z1";

    s = ToJsonString(tmp);

    std::cout << s << std::endl;

    return 0;
}
