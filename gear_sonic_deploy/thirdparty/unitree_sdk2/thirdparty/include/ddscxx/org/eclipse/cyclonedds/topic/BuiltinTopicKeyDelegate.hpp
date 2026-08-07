/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */


/**
 * @file
 */

#ifndef CYCLONEDDS_TOPIC_BUILTIN_TOPIC_KEY_DELEGATE_HPP_
#define CYCLONEDDS_TOPIC_BUILTIN_TOPIC_KEY_DELEGATE_HPP_

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{

class BuiltinTopicKeyDelegate
{
public:
    typedef uint32_t VALUE_T;
public:
    BuiltinTopicKeyDelegate() { }
    BuiltinTopicKeyDelegate(int32_t v[])
    {
        key_[0] = v[0];
        key_[1] = v[1];
        key_[2] = v[2];
    }
public:
    const int32_t* value() const
    {
        return key_;
    }

    void value(int32_t v[])
    {
        key_[0] = v[0];
        key_[1] = v[1];
        key_[2] = v[2];
    }

    bool operator ==(const BuiltinTopicKeyDelegate& other) const
    {
        return other.key_[0] == key_[0]
                 && other.key_[1] == key_[1]
                 && other.key_[2] == key_[2];
    }

private:
    int32_t key_[3];
};

}
}
}
}

#endif /* CYCLONEDDS_TOPIC_BUILTIN_TOPIC_KEY_DELEGATE_HPP_ */
