from threading import Thread
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, TextIteratorStreamer
import openai
import copy
import shutil
from PIL import Image
from argparse import ArgumentParser
import io
import pathlib
import gradio as gr
import time

import base64
import pathlib
from typing import Dict

import gradio as gr
import os
import time

from qwen_vl_utils import process_vision_info, smart_resize
import tempfile
import time
import imagesize
import uuid

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

image_transform = None
oss_reader = None


MAX_SEQ_LEN = 32000

DEFAULT_CKPT_PATH = 'Qwen/Qwen2-VL-7B-Instruct'

def compute_seqlen_estimated(tokenizer, json_input, sample_strategy_func):
    total_seq_len, img_seq_len, text_seq_len = 0, 0, 0
    for chat_block in json_input:

        chat_block['seq_len'] = 4
        role_length = len(tokenizer.tokenize(chat_block['role']))
        chat_block['seq_len'] += role_length
        text_seq_len += role_length

        for element in chat_block['content']:
            if 'image' in element:
                if 'width' not in element:
                    element['width'], element['height'] = imagesize.get(
                        element['image'].split('file://')[1])
                height, width = element['height'], element['width']
                height, width = sample_strategy_func(height, width)
                resized_height, resized_width = smart_resize(
                    height, width, max_pixels=14*14*4*5120)  # , min_pixels=14*14*4*512
                seq_len = resized_height * resized_width // 28 // 28 + 2  # add img_bos & img_eos
                element.update({
                    'resized_height': resized_height,
                    'resized_width': resized_width,
                    'seq_len': seq_len,
                })
                img_seq_len += element['seq_len']
                chat_block['seq_len'] += element['seq_len']
            elif 'video' in element:
                if isinstance(element['video'], (list, tuple)):
                    if 'width' not in element:
                        element['width'], element['height'] = imagesize.get(
                            element['video'][0].split('file://')[1])
                    height, width = element['height'], element['width']
                    height, width = sample_strategy_func(height, width)
                    resized_height, resized_width = smart_resize(
                        height, width, max_pixels=14*14*4*5120)  # , min_pixels=14*14*4*512
                    seq_len = (resized_height * resized_width // 28 // 28) * \
                        (len(element['video'])//2)+2  # add img_bos & img_eos
                    element.update({
                        'resized_height': resized_height,
                        'resized_width': resized_width,
                        'seq_len': seq_len,
                    })
                    img_seq_len += element['seq_len']
                    chat_block['seq_len'] += element['seq_len']
                else:
                    raise NotImplementedError
            elif 'text' in element:
                if 'seq_len' in element:
                    text_seq_len += element['seq_len']
                else:
                    element['seq_len'] = len(
                        tokenizer.tokenize(element['text']))
                    text_seq_len += element['seq_len']
                chat_block['seq_len'] += element['seq_len']
            elif 'prompt' in element:
                if 'seq_len' in element:
                    text_seq_len += element['seq_len']
                else:
                    element['seq_len'] = len(
                        tokenizer.tokenize(element['prompt']))
                    text_seq_len += element['seq_len']
                chat_block['seq_len'] += element['seq_len']
            else:
                raise ValueError('Unknown element: ' + str(element))
        total_seq_len += chat_block['seq_len']
    assert img_seq_len + text_seq_len + 4 * len(json_input) == total_seq_len
    total_seq_len += 1
    return {
        'content': json_input,
        'img_seq_len': img_seq_len,
        'text_seq_len': text_seq_len,
        'seq_len': total_seq_len,
    }


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int,
                        default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str,
                        default='127.0.0.1', help='Demo server name.')

    args = parser.parse_args()
    return args


def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        model = Qwen2VLForConditionalGeneration.from_pretrained(args.checkpoint_path,
                                                                torch_dtype='auto',
                                                                attn_implementation='flash_attention_2',
                                                                device_map=device_map)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path, device_map=device_map)

    processor = AutoProcessor.from_pretrained(args.checkpoint_path)
    return model, processor


class ChatSessionState:
    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.system_prompt: str = 'You are a helpful assistant.'
        self.model_name = ''
        self.image_cache = []


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)

        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)

    return transformed_messages


class Worker:
    def __init__(self):
        self.uids = []

        capture_image_dir = os.path.join("/tmp/captured_images")
        os.makedirs(capture_image_dir, exist_ok=True)
        self.capture_image_dir = capture_image_dir  # uid-to-messages

        self.save_dir = dict()
        self.messages = dict()  # uid-to-messages
        self.resized_width, self.resized_height = 640, 420
        # self.message_truncate = 0
        self.message_truncate = {}
        self.chat_session_states: Dict[str, ChatSessionState] = {}
        self.image_cache = {}

    def convert_image_to_base64(self, file_name):
        if file_name not in self.image_cache:
            self.image_cache[file_name] = {}
        if 'data_url' not in self.image_cache[file_name]:
            with open(file_name, 'rb') as f:
                self.image_cache[file_name]['data_url'] = 'data:image/png;base64,' + \
                    base64.b64encode(f.read()).decode('utf-8')
                assert self.image_cache[file_name]['data_url']
        return self.image_cache[file_name]['data_url']

    def get_session_state(self, session_id: str) -> ChatSessionState:
        """
        Retrieves the chat session state object for a given session ID.

        If the session ID does not exist in the currently managed session states,
        a new session state object is created and added to the list of managed sessions.

        Parameters:
        session_id: The unique identifier for the session.

        Returns:
        The session state object corresponding to the session ID.
        """
        # Check if the current session state collection already contains this session ID
        if session_id not in self.chat_session_states:
            # If it does not exist, create a new session state object and add it to the collection
            self.chat_session_states[session_id] = ChatSessionState(session_id)
        # Return the corresponding session state object
        return self.chat_session_states[session_id]

    def get_message_truncate(self, session_id):
        if session_id not in self.message_truncate:
            self.message_truncate[session_id] = 0
        return self.message_truncate[session_id]

    def truncate_messages_adaptive(self, messages):
        while True:
            seq_len = compute_seqlen_estimated(tokenizer, copy.deepcopy(
                messages), sample_strategy_func=lambda h, w: (h, w))['seq_len']
            if seq_len < MAX_SEQ_LEN:
                break
            # Remove the first block in content history:
            if len(messages[0]['content']) > 0 and 'video' in messages[0]['content'][0]:
                messages[0]['content'][0]['video'] = messages[0]['content'][0]['video'][2:]
                if len(messages[0]['content'][0]['video']) == 0:
                    messages[0]['content'] = messages[0]['content'][1:]
            else:
                messages[0]['content'] = messages[0]['content'][1:]

            # If the first block is empty, remove it:
            if len(messages[0]['content']) == 0:
                messages.pop(0)

            # If role is assistant, remove the first block in content history:
            if messages[0]['role'] == 'assistant':
                messages.pop(0)
        return messages

    def truncate_messages_by_count(self, messages, cnt):
        for i in range(cnt):
            # Remove the first block in content history:
            if len(messages[0]['content']) > 0 and 'video' in messages[0]['content'][0]:
                messages[0]['content'][0]['video'] = messages[0]['content'][0]['video'][2:]
                if len(messages[0]['content'][0]['video']) == 0:
                    messages[0]['content'] = messages[0]['content'][1:]
            else:
                messages[0]['content'] = messages[0]['content'][1:]

            # If the first block is empty, remove it:
            if len(messages[0]['content']) == 0:
                messages.pop(0)

            # If role is assistant, remove the first block in content history:
            if messages[0]['role'] == 'assistant':
                messages.pop(0)

    def get_save_dir(self, session_id):
        if self.save_dir.get(session_id) is None:
            temp_dir = tempfile.mkdtemp(dir=self.capture_image_dir)
            self.save_dir[session_id] = temp_dir
        return self.save_dir[session_id]

    def get_messages(self, session_id):
        if self.messages.get(session_id) is None:
            self.messages[session_id] = []
        return self.messages[session_id]

    def update_messages(self, session_id, role, content):
        if self.messages.get(session_id) is None:
            self.messages[session_id] = []
        messages = self.messages[session_id]
        if len(messages) == 0 or messages[-1]["role"] != role:
            messages.append({
                "role": role,
                "content": [content]
            })
        elif "video" in content and isinstance(content["video"], (list, tuple)) and "video" in messages[-1]["content"][-1] and isinstance(messages[-1]["content"][-1]["video"], (list, tuple)):
            messages[-1]["content"][-1]['video'].extend(content["video"])
        else:
            # If content and last message are all with type text, merge them
            if 'text' in messages[-1]["content"][-1] and 'text' in content:
                messages[-1]["content"][-1]['text'] += content["text"]
            else:
                messages[-1]["content"].append(content)
        self.messages[session_id] = messages

    def get_timestamp(self):
        return time.time()

    def chat(self, messages, request: gr.Request):
        messages = _transform_messages(messages)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs,
                           videos=video_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {'max_new_tokens': 512, 'streamer': streamer, **inputs}

        thread = Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def add_text(self, history, text, request: gr.Request):
        session_id = request.session_hash
        session_state: ChatSessionState = self.get_session_state(
            request.session_hash)

        if len(session_state.image_cache) > 0:
            for i, (timestamp, image_path) in enumerate(session_state.image_cache):
                if i % 2 == 0:
                    content = {"video": [f"file://{image_path}"]}
                else:
                    content["video"].append(f"file://{image_path}")
                    self.update_messages(
                        session_id, role="user", content=content)
                if i == len(session_state.image_cache)-1 and i % 2 == 0:
                    content["video"].append(content["video"][-1])
                    self.update_messages(
                        session_id, role="user", content=content)

            session_state.image_cache.clear()

        self.update_messages(session_id, role="user", content={
                             "type": "text", "text": text})

        history = history + [(text, None)]
        return history, ""

    def add_file(self, history, file, request: gr.Request):
        session_id = request.session_hash
        session_state: ChatSessionState = self.get_session_state(session_id)
        if isinstance(file, str) and file.startswith('data:'):
            # get binary bytes
            data = base64.b64decode(file.split('base64,')[1])
            # Create a file name using uuid
            filename = f'{uuid.uuid4()}.jpg'
            save_dir = self.get_save_dir(session_id)
            savename = os.path.join(save_dir, filename)
            # Save the file
            with open(savename, 'wb') as f:
                f.write(data)
            self.update_messages(session_id, role="user", content={
                                 "image": f"file://{savename}"})
        else:
            filename = os.path.basename(file.name)
            save_dir = self.get_save_dir(session_id)
            savename = os.path.join(save_dir, filename)
            if file.name.endswith('.mp4') or file.name.endswith('.mov'):
                shutil.copy(file.name, savename)
                os.makedirs(file.name + '.frames', exist_ok=True)
                os.system(
                    f'ffmpeg -i {file.name} -vf "scale=320:-1" -r 0.25 {file.name}.frames/%d.jpg')
                file_index = 1
                frame_list = []
                while True:
                    if os.path.isfile(os.path.join(f'{file.name}.frames/{file_index}.jpg')):
                        frame_list.append(os.path.join(
                            f'file://{file.name}.frames/{file_index}.jpg'))
                        file_index += 1
                    else:
                        break
                if len(frame_list) % 2 != 0:
                    frame_list = frame_list[1:]
                self.update_messages(session_id, role="user", content={
                                     "video": frame_list})
            else:
                shutil.copy(file.name, savename)
                self.update_messages(session_id, role="user", content={
                                     "image": f"file://{savename}"})

        history = history + [((savename,), None)]
        return history

    def add_image_to_streaming_cache(self, file, width, height, request: gr.Request):
        session_id = request.session_hash
        session_state: ChatSessionState = self.get_session_state(session_id)
        timestamp = self.get_timestamp()
        # If file is an image url starswith data:, save it to the session directory
        if isinstance(file, str) and file.startswith('data:'):
            # get binary bytes
            data = base64.b64decode(file.split('base64,')[1])
            width, height = int(width), int(height)
            # Load the image using PIL
            image = Image.open(io.BytesIO(data))
            # If width == -1, no need to scale the image
            if width == -1:
                pass
            else:
                # If height == -1, keep aspect ratio
                if height == -1:
                    height = round(width * image.height / float(image.width))
                image = image.resize((width, height), Image.LANCZOS)
            # Create a file name using uuid
            filename = f'{uuid.uuid4()}.jpg'
            save_dir = self.get_save_dir(session_id)
            savename = os.path.join(save_dir, filename)
            # Save the file
            image.save(savename, "JPEG")
        else:
            filename = os.path.basename(file.name)
            save_dir = self.get_save_dir(session_id)
            savename = os.path.join(save_dir, filename)
            shutil.copy(file.name, savename)

        session_state.image_cache.append((timestamp, savename))

    def response(self, chatbot_messages, request: gr.Request):
        session_id = request.session_hash
        messages = self.get_messages(session_id)
        self.truncate_messages_adaptive(messages)
        messages = copy.deepcopy(messages)
        chatbot_messages = copy.deepcopy(chatbot_messages)
        if chatbot_messages is None:
            chatbot_messages = []
        truncate_count = 0
        while True:
            compiled_messages = copy.deepcopy(messages)
            self.truncate_messages_by_count(
                compiled_messages, cnt=truncate_count)
            # Convert file:// image urls to data:base64 urls
            for message in compiled_messages:
                for content in message['content']:
                    if 'image' in content:
                        if content['image'].startswith('file://'):
                            content['image'] = self.convert_image_to_base64(
                                content['image'][7:])
                    elif 'video' in content and isinstance(content['video'], (list, tuple)):
                        for frame_i in range(len(content['video'])):
                            if content['video'][frame_i].startswith('file://'):
                                content['video'][frame_i] = self.convert_image_to_base64(
                                    content['video'][frame_i][7:])
            rep = self.chat(compiled_messages, request=request)
            try:
                for content in rep:
                    if not content:
                        continue
                    self.update_messages(session_id, role="assistant", content={
                                         "type": "text", "text": content})
                    if not chatbot_messages[-1][-1]:
                        chatbot_messages[-1][-1] = content
                    else:
                        chatbot_messages[-1][-1] += content
                    yield chatbot_messages
                break
            except openai.BadRequestError as e:
                print(e)
                if 'maximum context length' not in str(e):
                    raise e
                if self.messages[session_id][-1]['role'] == 'assistant':
                    chatbot_messages[-1][-1] = ''
                    self.messages[session_id] = self.messages[session_id][:-1]
                    # self.messages[session_id][-1]['content'][-1] = {'text': ''}
                self.message_truncate[session_id] += 1


recorder_js = pathlib.Path('recorder.js').read_text()
main_js = pathlib.Path('main.js').read_text()
GLOBAL_JS = pathlib.Path('global.js').read_text().replace('let recorder_js = null;', recorder_js).replace(
    'let main_js = null;', main_js)


def main():
    with gr.Blocks(js=GLOBAL_JS) as demo:
        gr.Markdown("""\
<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 80px"/><p>"""
                   )
        gr.Markdown("""<center><font size=8>Qwen2-VL</center>""")
        gr.Markdown("""\
<center><font size=3>This WebUI is based on Qwen2-VL, developed by Alibaba Cloud.</center>""")
        gr.Markdown("""<center><font size=3>本WebUI基于Qwen2-VL。</center>""")
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Accordion("System Prompt", open=False):
                textbox_system_prompt = gr.Textbox(
                    value="You are a helpful assistant.", label="System Prompt")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("Camera"):
                    image_camera = gr.Image(sources='webcam', label="Camera Preview",
                                            mirror_webcam=False, elem_id="gradio_image_camera_preview")
                    with gr.Accordion("Camera Settings", open=False):
                        with gr.Row():
                            camera_frame_interval = gr.Textbox(
                                "1", label="Frame interval or (1 / FPS)", elem_id="gradio_camera_frame_interval", interactive=True)
                        with gr.Row():
                            camera_width = gr.Textbox(
                                "640", label="Width (-1 = original resolution)")
                            camera_height = gr.Textbox(
                                "-1", label="Height (-1 = keep aspect ratio)")
                    with gr.Row():
                        button_camera_stream = gr.Button(
                            "Stream", elem_id="gradio_button_camera_stream")
                        button_camera_snapshot = gr.Button(
                            "Snapshot", elem_id="gradio_button_camera_snapshot")
                        button_camera_stream_submit = gr.Button(
                            "Snapshot", elem_id="gradio_button_camera_stream_submit", visible=False)
                with gr.Tab("Screen"):
                    image_screen = gr.Image(
                        sources='webcam', label="Screen Preview", elem_id="gradio_image_screen_preview")
                    with gr.Accordion("Screen Settings", open=False):
                        with gr.Row():
                            screen_frame_interval = gr.Textbox(
                                "5", label="Frame interval or (1 / FPS)", elem_id="gradio_screen_frame_interval", interactive=True)
                        with gr.Row():
                            screen_width = gr.Textbox(
                                "-1", label="Width (-1 = original resolution)")
                            screen_height = gr.Textbox(
                                "-1", label="Height (-1 = keep aspect ratio)")
                    with gr.Row():
                        button_screen_stream = gr.Button(
                            "Stream", elem_id="gradio_button_screen_stream")
                        button_screen_snapshot = gr.Button(
                            "Snapshot", elem_id="gradio_button_screen_snapshot")
                        button_screen_stream_submit = gr.Button(
                            "Snapshot", elem_id="gradio_button_screen_stream_submit", visible=False)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot([], elem_id="chatofa", height=500)
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter, or upload an image",
                        container=False,
                        scale=5,
                    )
                    btn = gr.UploadButton(
                        "📁", file_types=["image", "video", "audio"], scale=1)

                txt.submit(
                    fn=worker.add_text,
                    inputs=[chatbot, txt],
                    outputs=[chatbot, txt]
                ).then(
                    fn=worker.response,
                    inputs=[chatbot],
                    outputs=chatbot
                )

                btn.upload(
                    worker.add_file,
                    inputs=[chatbot, btn],
                    outputs=[chatbot]
                )

                # Camera
                button_camera_snapshot.click(
                    worker.add_file,
                    inputs=[chatbot, button_camera_snapshot],
                    outputs=[chatbot],
                    js="(p1, p2) => [p1, window.getCameraFrame()]",
                )
                button_camera_stream_submit.click(
                    worker.add_image_to_streaming_cache,
                    inputs=[button_camera_stream_submit,
                            camera_width, camera_height],
                    outputs=[],
                    js="(p1, p2, p3) => [window.getCameraFrame(), p2, p3]",
                )
                button_camera_stream.click(
                    lambda x: None,
                    inputs=[button_camera_stream],
                    outputs=[],
                    js="(p1, p2) => (window.startCameraStreaming())"
                )

                # Screen
                button_screen_snapshot.click(
                    worker.add_file,
                    inputs=[chatbot, button_screen_snapshot],
                    outputs=[chatbot],
                    js="(p1, p2) => [p1, window.getScreenshotFrame()]",
                )
                button_screen_stream_submit.click(
                    worker.add_image_to_streaming_cache,
                    inputs=[button_screen_stream_submit,
                            screen_width, screen_height],
                    outputs=[],
                    js="(p1, p2, p3) => [window.getScreenshotFrame(), p2, p3]",
                )
                button_screen_stream.click(
                    lambda x: None,
                    inputs=[button_screen_stream],
                    outputs=[],
                    js="(p1, p2) => (window.startScreenStreaming())"
                )
        with gr.Row():
            gr.Markdown("""\
    <font size=2>Note: This demo is governed by the original license of Qwen2-VL. \
    We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
    including hate speech, violence, pornography, deception, etc. \
    (注：本演示受Qwen2-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
    包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")
        demo.launch(
            share=args.share,
            inbrowser=args.inbrowser,
            server_port=args.server_port,
            server_name=args.server_name,
        )


if __name__ == '__main__':
    worker = Worker()
    args = _get_args()
    model, processor = _load_model_processor(args)
    tokenizer = processor.tokenizer
    main()
