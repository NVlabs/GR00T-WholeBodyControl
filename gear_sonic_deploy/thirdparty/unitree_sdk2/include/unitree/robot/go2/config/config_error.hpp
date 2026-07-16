#ifndef __UT_ROBOT_GO2_CONFIG_ERROR_HPP__
#define __UT_ROBOT_GO2_CONFIG_ERROR_HPP__

namespace unitree
{
namespace robot
{
namespace go2
{
UT_DECL_ERR(UT_CONFIG_ERR_PARAMETER,        8201,   "parameter error.")
UT_DECL_ERR(UT_CONFIG_ERR_NOT_FOUND,        8202,   "config name is not found.")
UT_DECL_ERR(UT_CONFIG_ERR_NAME,             8203,   "name is invalid.")
UT_DECL_ERR(UT_CONFIG_ERR_LIMITED,          8204,   "name/content length limited.")
UT_DECL_ERR(UT_CONFIG_ERR_LOCK,             8205,   "lock error.")
UT_DECL_ERR(UT_CONFIG_ERR_LOAD_META,        8206,   "load meta error.")
UT_DECL_ERR(UT_CONFIG_ERR_SAVE_META,        8207,   "save meta error.")
UT_DECL_ERR(UT_CONFIG_ERR_TEMP_META,        8208,   "save meta temp error.")
UT_DECL_ERR(UT_CONFIG_ERR_FORMAL_META,      8209,   "formalize meta error.")
UT_DECL_ERR(UT_CONFIG_ERR_REMOVE_META,      8210,   "remove meta error.")
UT_DECL_ERR(UT_CONFIG_ERR_LOAD_DATA,        8211,   "load data error.")
UT_DECL_ERR(UT_CONFIG_ERR_SAVE_DATA,        8212,   "save data error.")
UT_DECL_ERR(UT_CONFIG_ERR_TEMP_DATA,        8213,   "save data temp error.")
UT_DECL_ERR(UT_CONFIG_ERR_FORMAL_DATA,      8214,   "formalize data error.")
UT_DECL_ERR(UT_CONFIG_ERR_REMOVE_DATA,      8215,   "remove data error.")
}
}
}

#endif//__UT_ROBOT_GO2_CONFIG_ERROR_HPP__
