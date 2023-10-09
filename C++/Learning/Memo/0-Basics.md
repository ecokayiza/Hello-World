### 1.compile

- compile---将高级语言文本 (source file) 转换为二进制机器码(object file)
- linking---将多个obj file合并形成可执行文件(executable file)
``.cpp >> .obj >> .exe`` 
*link目标包括project中的sourcefile和dependencies,可使内部和库中的函数、数据互相调用*
### 2.library
- library---提前编辑好的object file,通常是常用的指令
- *使用外部库时要包括头文件和库文件* `.h` `.lib`
- 并在linker中设置好调用路径
- `lib`包括静态库和动态库； 静态库直接载程序，动态库只在运行时调用
