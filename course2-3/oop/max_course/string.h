#pragma once

#include <ostream>
#include "container.h"

class String : public SequenceContainer<const char> {
public:
    String(const char []);

    String();

    String &operator=(const char arr[]);

    friend std::ostream &operator<<(std::ostream &out, const String &str);

};
