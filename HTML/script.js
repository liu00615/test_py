const featureMethods = {
    "searchbycolor": ["Color_Histogram", "HSV_Moments"],
    "searchbyedge": ["HOG_Features", "Hu_Moments"],
    "searchbytexture": ["GLCM", "LBP"],
    "searchbysift": ["SIFT"],
    "searchbyvggkeras": ["VGG16"],
    "searchbyhash": ["aHash", "dHash", "pHash"]
};

// 点击搜索方式按钮时，标记为选中状态
document.getElementById("search-buttons").addEventListener("click", function(event) {
    if (event.target.classList.contains("method-btn")) {
        // 移除之前选中的按钮
        const selectedBtn = document.querySelector("#search-buttons .method-btn.selected");
        if (selectedBtn) {
            selectedBtn.classList.remove("selected");
        }

        // 标记当前按钮为选中
        event.target.classList.add("selected");

        var searchMethod = event.target.getAttribute("data-method");
        updateFeatureButtons(searchMethod);  // 更新特征方法按钮
    }
});

// 动态更新特征方法按钮
function updateFeatureButtons(searchMethod) {
    var featureButtonsContainer = document.getElementById("feature-buttons");
    featureButtonsContainer.innerHTML = ""; // 清空之前的按钮

    if (featureMethods[searchMethod]) {
        featureMethods[searchMethod].forEach(function(method) {
            var button = document.createElement("button");
            button.classList.add("method-btn");
            button.textContent = method;
            button.setAttribute("data-method", method);
            
            // 添加点击事件标记选中
            button.addEventListener("click", function() {
                const selectedBtn = document.querySelector("#feature-buttons .method-btn.selected");
                if (selectedBtn) {
                    selectedBtn.classList.remove("selected");
                }
                button.classList.add("selected");
            });

            featureButtonsContainer.appendChild(button);
        });
        featureButtonsContainer.disabled = false; // 启用特征方法按钮组
    } else {
        featureButtonsContainer.disabled = true; // 禁用特征方法按钮组
    }
}

// 展示选择的图片
function previewImage() {
    var fileInput = document.getElementById("file-upload");
    var file = fileInput.files[0];

    if (file) {
        var reader = new FileReader();
        reader.onload = function(e) {
            var imgElement = document.createElement("img");
            imgElement.src = e.target.result;
            imgElement.alt = "查询图片";

            var imageContainer = document.getElementById("uploaded-image");
            imageContainer.innerHTML = "<h2>原始查询图像</h2>"; // 清空之前的内容并添加标题
            imageContainer.appendChild(imgElement);
        };
        reader.readAsDataURL(file);  // 读取文件内容并展示
    }
}

// 提交图片到后端进行检索
function submitImage() {
    var searchMethod = document.querySelector("#search-buttons .method-btn.selected")?.getAttribute("data-method");
    var featureMethod = document.querySelector("#feature-buttons .method-btn.selected")?.getAttribute("data-method");
    var fileInput = document.getElementById("file-upload");
    var file = fileInput.files[0];

    // 提示用户选择搜索方法和特征方法
    var errorMessage = document.getElementById("error-message");
    errorMessage.innerHTML = ""; // 清空错误信息

    if (!searchMethod) {
        alert("请先选择搜索方式");
        return;
    }

    if (!featureMethod) {
        alert("请先选择特征方法");
        return;
    }

    if (!file) {
        alert("请先选择查询图片");
        return;
    }

    var formData = new FormData();
    formData.append("file", file);

    // 发送图片到指定的API
    fetch(`http://127.0.0.1:5000/${searchMethod}`, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data, featureMethod); // 传入特征方法，过滤结果
    })
    .catch(error => {
        console.error("请求失败:", error);
        alert("请求失败，请检查后端是否运行");
    });
}

// 展示返回的检索结果
function displayResults(data, featureMethod) {
    var resultContainer = document.getElementById("result");
    resultContainer.innerHTML = "<h2>检索结果</h2>";  // 清空之前的结果并添加标题

    // 遍历每种方法的结果
    for (let method in data) {
        // 只有用户选择的特征方法对应的结果会被展示
        if (method === featureMethod && data[method].length > 0) {
            var methodGroup = document.createElement("div");
            methodGroup.classList.add("method-group");

            var methodTitle = document.createElement("h3");
            methodTitle.innerText = `${method} 方法`;
            methodGroup.appendChild(methodTitle);

            var imageRow = document.createElement("div");
            imageRow.classList.add("image-row");

            data[method].forEach(item => {
                var img = document.createElement("img");
                var imgPath = `../data/256_ObjectCategories/${item.image}`;
                img.src = imgPath;
                img.alt = item.image;
                img.title = `相似度: ${item.similarity}`;

                // 格式化相似度为百分比形式，保留五位小数
                var similarityPercent = (item.similarity * 100).toFixed(3) + '%';

                // 创建相似度显示元素
                var imageInfo = document.createElement("div");
                imageInfo.classList.add("image-info");
                imageInfo.innerText = `相似度: ${similarityPercent}`;

                // 将图片和相似度显示组合在一起
                var imageContainer = document.createElement("div");
                imageContainer.appendChild(img);
                imageContainer.appendChild(imageInfo);

                imageRow.appendChild(imageContainer);
            });

            methodGroup.appendChild(imageRow);
            resultContainer.appendChild(methodGroup);
        }
    }
}
