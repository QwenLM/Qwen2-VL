// Setup if needed and start recording.
async () => {
    screenButton = document.getElementById('gradio_image_screen_preview'); //.querySelector('button');
    //eventListers = window.getEventListeners(screenButton);
    //screenButton.removeEventListener('click', eventListers['click'][0].listener, eventListers['click'][0].useCapture)
    // If window.getScreenshotFrameDoes not exist:
    if (!window.getScreenshotFrame) {
        // Define the function
        window.getScreenshotFrame = () => {
            // Get the video element
            var video = document.getElementById('gradio_image_screen_preview').querySelector('video');
            // Get the canvas element
            var canvas = document.getElementsByTagName('canvas')[0];
            // Get the canvas context
            var ctx = canvas.getContext('2d');
            // Set the canvas size to match the video size
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Draw the current frame on the canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', 1.0);
        }
    }

    if (!window.getCameraFrame) {
        // Define the function
        window.getCameraFrame = () => {
            // Get the video element
            var video = document.getElementById('gradio_image_camera_preview').querySelector('video');
            // Get the canvas element
            var canvas = document.getElementsByTagName('canvas')[0];
            // Get the canvas context
            var ctx = canvas.getContext('2d');
            // Set the canvas size to match the video size
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            // Draw the current frame on the canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL('image/jpeg', 1.0);
        }
    }

    if(!window.startCameraStreaming) {
        window.startCameraStreaming = () => {
            var intervalString = document.getElementById("gradio_camera_frame_interval").getElementsByTagName('textarea')[0].value;
            var interval = parseFloat(intervalString) * 1000;
            console.log("Start camera, interval: " + interval + "")
            window.cameraIntervalId = setInterval(() => document.getElementById('gradio_button_camera_stream_submit').click(), interval)
        }
    }

    if(!window.startScreenStreaming) {
        window.startScreenStreaming = () => {
            var intervalString = document.getElementById("gradio_screen_frame_interval").getElementsByTagName('textarea')[0].value;
            var interval = parseFloat(intervalString) * 1000;
            console.log("Start screen, interval: " + interval + "")
            window.screenIntervalId = setInterval(() => document.getElementById('gradio_button_screen_stream_submit').click(), interval)
        }
    }

    if(document.getElementsByTagName('canvas').length <= 0) {
        var canvasElement = document.createElement('canvas');
        canvasElement.style.position = 'fixed';
        canvasElement.style.bottom = '0px';
        canvasElement.style.right = '0px';
        canvasElement.style.width = '32px'; // 您可以根据需要调整宽度
        canvasElement.style.height = '32px'; // 高度自适应，保持宽高比
        document.body.appendChild(canvasElement);
    }

    screenButton.addEventListener('click', function (e) {
        //alert('Hello world! 666')

        // Set up recording functions if not already initialized
        if (!window.startRecording) {
            let recorder_js = null;
            let main_js = null;
        }

        if (!window.getVideoSnpapshot) {
            // Synchronous function to get a video snapshot
            window.getVideoSnpapshot = () => {
                var canvas = document.getElementsByTagName('canvas')[0];
                var ctx = canvas.getContext('2d');
                // window.getComputedStyle(canvas)
                if(canvas.width != canvas.clientWidth) {
                    canvas.width = canvas.clientWidth
                }
                if(canvas.height != canvas.clientHeight) {
                    canvas.height = canvas.clientHeight
                }
                if(!window.videoPlaying) {
                    return "Record";
                }
                ctx.drawImage(document.getElementById('video_screenshot'), 0, 0, canvas.clientWidth, canvas.clientHeight);
                console.log(canvas.toDataURL('image/jpeg', 1.0));
                return canvas.toDataURL('image/jpeg', 1.0);
            };
        }
        e.stopPropagation();
        window.startRecording();
    }, true)
}