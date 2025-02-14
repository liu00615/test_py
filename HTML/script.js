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
    var fileInput = document.getElementById("file-upload");
    var file = fileInput.files[0];

    if (!file) {
        alert("请先选择一张图片");
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
        displayResults(data);
    })
    .catch(error => {
        console.error("请求失败:", error);
        alert("请求失败，请检查后端是否运行");
    });
}

// 展示返回的检索结果
function displayResults(data) {
    var resultContainer = document.getElementById("result");
    resultContainer.innerHTML = "<h2>检索结果</h2>";  // 清空之前的结果并添加标题

    // 遍历每种方法的结果
    for (let method in data) {
        if (data[method].length > 0) {
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

                var similarityText = document.createElement("div");
                similarityText.classList.add("similarity");
                similarityText.innerText = `相似度: ${item.similarity.toFixed(4)}`;

                // 包裹图片和相似度文本
                var imgContainer = document.createElement("div");
                imgContainer.classList.add("image-container");
                imgContainer.appendChild(img);
                imgContainer.appendChild(similarityText);

                imageRow.appendChild(imgContainer);
            });

            methodGroup.appendChild(imageRow);
            resultContainer.appendChild(methodGroup);
        }
    }
}
