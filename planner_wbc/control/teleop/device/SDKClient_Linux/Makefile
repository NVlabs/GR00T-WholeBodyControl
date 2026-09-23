CXX      := g++

#CXXFLAGS := -pedantic-errors -Wall -Wextra -Werror -g --std=c++17
# CXXFLAGS := -g --std=c++17

PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYBIND11_SUFFIX := $(shell python3-config --extension-suffix)
CXXFLAGS := -shared -g --std=c++17 -fPIC ${PYBIND11_INCLUDES}

LDFLAGS  := -L./ManusSDK/lib -lManusSDK -lncurses -lpthread -Wl,-rpath=./ManusSDK/lib
BUILD    := ./
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)

# TARGET   := SDKClient_Linux.out
TARGET   := ManusServer$(PYBIND11_SUFFIX)

INCLUDE  := -I./ManusSDK/include
SRC      :=  \
   $(wildcard *.cpp)

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)
	
-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -g
release: all

clean:
	-@rm $(TARGET)
	-@rm -rvf $(OBJ_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"
