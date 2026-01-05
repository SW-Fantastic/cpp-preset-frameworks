# Cpp presets frameworks

Cpp-preset的Framework，可以方便的在Java使用我的CppPreset项目。

## CppPreset项目

基于JavaCPP封装的C++ Native库，可以方便的在Java中进行使用，是本项目的基础。

> CppPreset项目链接： [CppPreset project](https://github.com/SW-Fantastic/cpp-presets)

## 子项目列表（sub-project list）

1. Live2D Core & Live2D Java SDK（YES，**JAVA SDK，NOT ANDROID**）
    - Build with Live2D Native SDK version 5-r4.1
    - Has pre-built binary with (windows-x64, linux-x64, macos-x64)
      本库提供了预构建的Windows，linux，macos的64位二进制文件。
2. Pdfium Core & Pdfium4J
    - Build with pdfium version 122.0.6248
    - Has pre-built binary with (window-x64, linux-x64, ~~macos-x64~~)
      本库提供了预构建的Windows，linux和~~macos~~的64位二进制文件。
3. MariaDB Embedded & MariaDB Embedded JDBC（Embedded Mariadb java version）
    - Build with Mariadb version 11.6
    - Has pre-built binary with (windows-x64, linux-x64)
      本库提供了预构建的Windows，linux的64位二进制文件。
    - this library has dependency with repository `our-commons`，Please build and install
      it before you do build of this one，本库依赖了另一个`our-commons`库。
    - `our-commons` [Click here for this repository](https://github.com/SW-Fantastic/our-commons)
4. DearImGui（Library only，no frameworks at this time）
    - Build with DearImGUI Docking branch version 1.91.1
    - Has pre-built binary with (windows-x64)，目前只有Windows，macos没有更新。
5. LLama.cpp (framework is developing, not release yet)
    - Build with llama.cpp version b4730.
    - Has pre-built binary with (windows-x64 cpu only)