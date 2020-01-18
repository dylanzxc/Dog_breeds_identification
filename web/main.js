let url = {
    'vgg': 'http://34.83.137.89:5000/vgg',
    'resnet': 'http://34.82.59.181:5000/resnet',
    'inception': 'http://34.82.246.70:5000/inception'
}

// let url = {
//     'vgg': 'http://vggapi.eutial.com:5000/vgg',
//     'resnet': 'http://resnetapi.eutial.com:5000/resnet',
//     'inception': 'http://inceptionapi.eutial.com:5000/inception'
// };
let hasImage = false;

window.onload = async function () {
    let toptip = document.getElementById('top-tip');
    try {
        const response = await fetch('http://34.83.137.89:5000/init', {
            method: 'GET',
            mode: 'cors',
            headers: {
                'Access-Control-Allow-Origin': '*'
            }
        });
        const text = await response.text();
        toptip.innerHTML = text;
    } catch (error) {
        console.log(`Init failed: ${error}`);
    };
};

async function postData(dst, data) {
    const response = await fetch(dst, {
        method: 'POST',
        body: data,
        mode: 'cors',
        headers: {
            'Access-Control-Allow-Origin': '*'
        }
    });
    return await response.text();
};

function sendImage() {
    let checkedValue = document.querySelector('.form-check-input:checked').value;
    let formdata = new FormData();
    let filesection = document.getElementById('file-section');
    let resultarea = document.getElementById('result-area');
    resultarea.innerHTML = '';
    formdata.append('image', filesection.files[0]);
    postData(url[checkedValue], formdata).then(value => dataProcess(value));
};

function showPreview(e) {
    let preview = document.getElementById('preview');
    let filereader = new FileReader();
    if (e.files[0]) {
        filereader.onload = () => preview.innerHTML = '<img class="img-thumbnail animated fadeIn" src="' + filereader.result + '" />';
        filereader.readAsDataURL(e.files[0]);
        hasImage = true;
    };

    if (hasImage) {
        let uploadbutton = document.getElementById('upload-button');
        uploadbutton.style.display = 'block';
    };
};

function dataProcess(raw_data) {
    let parsed_data = JSON.parse(raw_data);
    let resultarea = document.getElementById('result-area');
    for (let i of parsed_data.predictions) {
        let label = i.label;
        let prob = i.probability.toFixed(2) * 100
        resultarea.innerHTML += label + ': ' + prob + '%' + '<br>'
    }
}