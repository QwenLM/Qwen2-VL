// recorder.js
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const BLOB_TYPE = "video/webm";
class ScreenCastRecorder {
    /** True if the current browser likely supports screencasts. */
    static isSupportedBrowser() {
        return (navigator.mediaDevices != null &&
            navigator.mediaDevices.getUserMedia != null &&
            navigator.mediaDevices.getDisplayMedia != null &&
            MediaRecorder.isTypeSupported(BLOB_TYPE));
    }
    constructor({ recordAudio, onErrorOrStop }) {
        this.recordAudio = recordAudio;
        this.onErrorOrStopCallback = onErrorOrStop;
        this.inputStream = null;
        this.recordedChunks = [];
        this.mediaRecorder = null;
    }
    /**
     * This asynchronous method will initialize the screen recording object asking
     * for permissions to the user which are needed to start recording.
     */
    initialize() {
        return __awaiter(this, void 0, void 0, function* () {
            const desktopStream = yield navigator.mediaDevices.getDisplayMedia({
                video: true,
            });
            let tracks = desktopStream.getTracks();
            if (this.recordAudio) {
                const voiceStream = yield navigator.mediaDevices.getUserMedia({
                    video: false,
                    audio: true,
                });
                tracks = tracks.concat(voiceStream.getAudioTracks());
            }
            this.recordedChunks = [];
            this.inputStream = new MediaStream(tracks);
            let videoElement = document.getElementById('gradio_image_screen_preview').querySelectorAll('video')[0]
            // Remove hide class from videoElement
            videoElement.classList.remove('hide');
            videoElement.classList.remove('flip');

            // Set src to inputStream
            videoElement.srcObject = this.inputStream;
            window.screenInputStream = this.inputStream;
            videoElement.play();

            // Get width and height of inputStream
            //window.videoElement.width = this.inputStream.getVideoTracks()[0].getSettings().width;
            window.videoPlaying = 1;
            /*setInterval(() => {
                document.getElementById("component-2").click()
            }, 5000);*/
            this.mediaRecorder = new MediaRecorder(this.inputStream, {
                mimeType: BLOB_TYPE,
            });
            this.mediaRecorder.ondataavailable = e => this.recordedChunks.push(e.data);
        });
    }
    getState() {
        if (this.mediaRecorder) {
            return this.mediaRecorder.state;
        }
        return "inactive";
    }
    /**
     * This method will start the screen recording if the user has granted permissions
     * and the mediaRecorder has been initialized
     *
     * @returns {boolean}
     */
    start() {
        if (!this.mediaRecorder) {
            console.warn(`ScreenCastRecorder.start: mediaRecorder is null`);
            return false;
        }
        const logRecorderError = (e) => {
            console.warn(`mediaRecorder.start threw an error: ${e}`);
        };
        this.mediaRecorder.onerror = (e) => {
            logRecorderError(e);
            this.onErrorOrStopCallback();
        };
        this.mediaRecorder.onstop = () => this.onErrorOrStopCallback();
        try {
            this.mediaRecorder.start();
        }
        catch (e) {
            logRecorderError(e);
            return false;
        }
        return true;
    }
    /**
     * This method will stop recording and then return the generated Blob
     *
     * @returns {(Promise|undefined)}
     *  A Promise which will return the generated Blob
     *  Undefined if the MediaRecorder could not initialize
     */
    stop() {
        if (!this.mediaRecorder) {
            return undefined;
        }
        let resolver;
        const promise = new Promise(r => {
            resolver = r;
        });
        this.mediaRecorder.onstop = () => resolver();
        this.mediaRecorder.stop();
        if (this.inputStream) {
            this.inputStream.getTracks().forEach(s => s.stop());
            this.inputStream = null;
        }
        return promise.then(() => this.buildOutputBlob());
    }
    buildOutputBlob() {
        return new Blob(this.recordedChunks, { type: BLOB_TYPE });
    }
}
