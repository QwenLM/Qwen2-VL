// main.js
if (!ScreenCastRecorder.isSupportedBrowser()) {
    console.error("Screen Recording not supported in this browser");
}
let recorder;
let outputBlob;
const stopRecording = () => __awaiter(void 0, void 0, void 0, function* () {
    let currentState = "RECORDING";
    // We should do nothing if the user try to stop recording when it is not started
    if (currentState === "OFF" || recorder == null) {
        return;
    }
    // if (currentState === "COUNTDOWN") {
    //     this.setState({
    //         currentState: "OFF",
    //     })
    // }
    if (currentState === "RECORDING") {
        if (recorder.getState() === "inactive") {
            // this.setState({
            //     currentState: "OFF",
            // })
            console.log("Inactive");
        }
        else {
            outputBlob = yield recorder.stop();
            console.log("Done recording");
            // this.setState({
            //   outputBlob,
            //   currentState: "PREVIEW_FILE",
            // })
            window.currentState = "PREVIEW_FILE";
            const videoSource = URL.createObjectURL(outputBlob);
            window.videoSource = videoSource;
            const fileName = "recording";
            const link = document.createElement("a");
            link.setAttribute("href", videoSource);
            link.setAttribute("download", `${fileName}.webm`);
            link.click();
        }
    }
});
const startRecording = () => __awaiter(void 0, void 0, void 0, function* () {
    const recordAudio = true;
    recorder = new ScreenCastRecorder({
        recordAudio,
        onErrorOrStop: () => stopRecording(),
    });
    try {
        yield recorder.initialize();
    }
    catch (e) {
        console.warn(`ScreenCastRecorder.initialize error: ${e}`);
        //   this.setState({ currentState: "UNSUPPORTED" })
        window.currentState = "UNSUPPORTED";
        return;
    }
    // this.setState({ currentState: "COUNTDOWN" })
    const hasStarted = recorder.start();
    if (hasStarted) {
        // this.setState({
        //     currentState: "RECORDING",
        // })
        console.log("Started recording");
        window.currentState = "RECORDING";
    }
    else {
        stopRecording().catch(err => console.warn(`withScreencast.stopRecording threw an error: ${err}`));
    }
});

// Set global functions to window.
window.startRecording = startRecording;
window.stopRecording = stopRecording;