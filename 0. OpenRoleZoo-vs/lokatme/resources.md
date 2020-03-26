# 资源编译工具说明

资源编译工具，可以把资源文件编译到CPP源文件，CPP模式支持10个文件，
共可以支持`1G`的资源文件的编译。

## 工具说明
```
Usage: command [option] input_path
    Input path can be file or folder
Option:
    [--out_dir -o]: set generated files output dir
    [--in_dir -i]: set resources files input root dir
    [--filename --fn -n]: set output filename
    [--help -? -h]: print help documents
    [--cpp -cpp]: use cpp mode
```
当打开`-cpp`模式时， 则使用C++模式编译出C++源文件。
`input_path`指向资源文件，或者资源文件夹。

## 资源文件格式
资源文件扩展名为`orc`。
以下是示例资源文件文件内容：
```
# Add every line to the format /url:path.

/model:binary.bin
```
其中`#`开头的行为注释行，所有非注释行为资源描述。
资源描述必须包含`:`，`:`之前为资源标志符，`:`之后是资源路径。
资源文件名不可以包含`:`，资源路径的根目录为`--in_dir`路径，或者资源文件路径。

[注意]：
当没有资源文件，并且`input_path`为文件夹时，
会自动把文件夹内的所有文件编译，资源标志符为`/`加上资源路径。

## C++ 模式

### 资源获取接口
编译会生成头文件如下，通过调用`orz_resource_get`接口，凭借资源描述符来获取文件内容。
[注意]：资源描述符开头如果存在的第一个`@`符号会被忽略，可以用于自动判别是资源文件还是实际文件。
头文件名为`orz_cpp_resources.h`：
```cpp
#ifndef _INC_ORZ_CPP_RESOURCES_AUTO_COMPILATION_H
#define _INC_ORZ_CPP_RESOURCES_AUTO_COMPILATION_H


#include <string>
#include <memory>

/**
 * \brief ORZ resources structure
 */
class orz_resources {
public:
    orz_resources() = default;
    explicit orz_resources(size_t size) : size(size) {
        data.reset(new char[size], std::default_delete<char[]>());
    }

    std::shared_ptr<char> data;    ///< memory pointer to the resource
    size_t size = 0;               ///< size of the resource
};

#ifdef _MSC_VER
#define ORZ_RESOURCES_HIDDEN_API
#else
#define ORZ_RESOURCES_HIDDEN_API __attribute__((visibility("hidden")))
#endif

/**
 * \brief Get ORZ resource by URL
 * \param url The URL described in the orc file
 * \return return \c `struct orz_resources`
 * \note Return { NULL, 0 } if failed.
 * \note It will ignore the symbol `@` at the beginning of the string.
 */
ORZ_RESOURCES_HIDDEN_API
const orz_resources orz_resources_get(const std::string &url);


#endif //_INC_ORZ_CPP_RESOURCES_AUTO_COMPILATION_H
```

### 生成的源文件

编译会生成源码为`orz_cpp_resources.cpp` 和 `orz_cpp_resources.[0-9]*.cpp`，
其中`$1` 为0到9。

所有的源文件都应该被编译进最终的文件。

[注意]：单个项目只能有一个资源文件，否则会出现命名冲突。

### 工程支持

**MSVC**
`Vistual studio`支持提供了资源文件`ORZCppResources.props`。
把属性表加入项目，在跟目录文件`orz_resources.orc`中编辑资源文件，
并使用`orz_resource_get`接口就可以获取资源文件了。

**CMake**
`CMake`工程要添加`ORZCppResources.cmake`，具体使用代码如下：
```cmake
include(ORZCppResources)
add_orz_resources(${PATH_TO_ROOT_OF_ORC} HEADER_FILES SOURCE_FILES)
```
使用时要把`${SOURCE_FILES}`加入到源代码中。

**Makefile**
不提供工具，可以根据生成文件的协议，来自己构建编译流程。

## 纯C 模式
已经弃用，只要不打开`-cpp`选项，编译出的就是纯c源代码，
支持的资源文件总大小不超过`128M`。