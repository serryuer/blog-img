code snippet用于方便代码编写人员快速输入常用代码片段或者模板，vscode在2019年的版本更新中加强了这一特性。

Vscode的Code snippet是根据不同语言配置的，每一个语言有一个$language.json$文件用于配置该语言所关联的文件类型编写时可用的代码片段，如下图所示：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/vscode-snippet/Snipaste_2019-04-19_09-27-27.jpg" width=50% height=50%/>


<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/vscode-snippet/Snipaste_2019-04-19_09-45-58.jpg" width=50% height=50%/>

这个json文件的配置方式网上已经有很多解释，具体请参考[Creating your own snippets](https://code.visualstudio.com/docs/editor/userdefinedsnippets)

除了根据文件类型配置代码片段之外，还可以配置全局的代码片段，或者为工作空间配置特定的代码片段，需要新建一个文件，如果是为工作空间新建代码片段，一般放在工作空间下的.vscode文件夹下，文件名为$***.code-snippets$，语法和上面是一样的。

但是据笔者实验，只是上面的设置并不能实现Markdown文件编写时代码片段的快捷使用，我们还需要针对markdown类型文件进行具体的设置，打开$settings.json$文件，添加如下配置：
```
"[markdown]": {
    "editor.formatOnSave": true,
    "editor.renderWhitespace": "all",
    "editor.quickSuggestions": {
        "other": true,
        "comments": true,
        "strings": true
    },
    "editor.acceptSuggestionOnEnter": "off"
}
```

效果如下：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/vscode-snippet/Snipaste_2019-04-19_09-50-23.jpg" width=50% height=50%/>

