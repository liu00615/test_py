const featureMethods = {
    "searchbycolor": ["Color_Histogram", "HSV_Moments"],
    "searchbyedge": ["HOG_Features", "Hu_Moments"],
    "searchbytexture": ["GLCM", "LBP"],
    "searchbysift": ["SIFT"],
    "searchbyvggkeras": ["VGG16"],
    "searchbyhash": ["aHash", "dHash", "pHash"]
};

// 动态更新特征方法选择框
document.getElementById("search-method").addEventListener("change", function() {
    var searchMethod = this.value;
    var featureSelect = document.getElementById("feature-method");

    // 清空并启用特征方法选择框
    featureSelect.innerHTML = '<option value="" disabled selected>请选择特征方法</option>';
    if (searchMethod && featureMethods[searchMethod]) {
        featureMethods[searchMethod].forEach(function(method) {
            var option = document.createElement("option");
            option.value = method;
            option.textContent = method;
            featureSelect.appendChild(option);
        });

        featureSelect.disabled = false; // 启用特征方法选择框
    } else {
        featureSelect.disabled = true; // 禁用特征方法选择框
    }
});

// 页面加载时不默认选择任何选项
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById("search-method").selectedIndex = -1; // 取消默认选择
    document.getElementById("feature-method").disabled = true; // 禁用特征方法选择框
});

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
    var searchMethod = document.getElementById("search-method").value;
    var featureMethod = document.getElementById("feature-method").value;
    var fileInput = document.getElementById("file-upload");
    var file = fileInput.files[0];

    // 提示用户选择搜索方法和特征方法
    var errorMessage = document.getElementById("error-message");
    errorMessage.innerHTML = ""; // 清空错误信息

    if (!searchMethod) {
        errorMessage.innerHTML = "请先选择搜索方式";
        return;
    }

    if (!featureMethod) {
        errorMessage.innerHTML = "请先选择特征方法";
        return;
    }

    if (!file) {
        errorMessage.innerHTML = "请先选择查询图片";
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

                imageRow.appendChild(img);
            });

            methodGroup.appendChild(imageRow);
            resultContainer.appendChild(methodGroup);
        }
    }
}
