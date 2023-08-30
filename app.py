from flask import Flask,request,jsonify
import os,json,cv2,random
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips, ImageClip, CompositeVideoClip
from PIL import ImageFont, ImageDraw, Image
from threading import Thread

app = Flask(__name__)
@app.route('/',methods=['POST'])
def index():
    id=new_video_id()
    data=request.json
    ideas_file = data.get('ideas_file')
    lang = data.get('lang')
    audio_folder = data.get('audio_folder')
    if not os.path.exists(audio_folder):
        return jsonify({"error":"audio folder doesn't exist."})
    image_paths = data.get('image_paths')
    print(image_paths)
    fps = data.get('fps', 24)
    text_color = tuple(data.get('text_color', (0, 0, 0)))
    text_bg_color = tuple(data.get('text_bg_color', (255, 200, 0)))
    estimated_duration=makeVideo(id,ideas_file, lang, audio_folder, image_paths, fps, text_color, text_bg_color)
    if estimated_duration<0:
        return jsonify({"error" : f"Audio file {-estimated_duration-1}.mp3 does not exist."})
    return jsonify({"video id":id ,
                    "estimated duration (in seconds)" : estimated_duration})

@app.route('/<int:target_id>')
def get_id_status__phase(target_id):
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)

    if str(target_id) in data:
        status = data[str(target_id)]["status"]
        if status=="Not ready" :
            phase = data[str(target_id)]["phase_out_of_5"]
            return jsonify({"video status" : status,
                            "phase" : f"{phase}/5"})
        else :
            current_directory = os.path.dirname(os.path.abspath(__file__))
            p = os.path.join(current_directory, f"{target_id}.mp4")
            return jsonify({"video status" : status,
                            "video path" : p})
    else:
        return jsonify({"error" : f"ID {target_id} not found."})

def new_video_id():
    json_file_path = 'data.json'
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as json_file:
            initial_data = {'last_id': 999}
            json.dump(initial_data, json_file)

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    new_id = data['last_id'] + 1

    data['last_id'] = new_id
    data[str(new_id)] = {
        "status": "Not ready",
        "phase_out_of_5": 1
    }

    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    return new_id

def update_video_status(target_id, new_status):
    json_file_path = 'data.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    if str(target_id) in data:
        data[str(target_id)]["status"] = new_status
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

def update_video_phase(target_id, new_phase):
    json_file_path = 'data.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    if str(target_id) in data:
        data[str(target_id)]["phase_out_of_5"] = new_phase
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

def makeVideo(id,ideas_file, lang, audio_folder, image_paths, fps, text_color, text_bg_color):
    with open(ideas_file,'r') as file:
        ideas=file.readlines()
    for i in range(len(ideas)):
        if(i!=len(ideas)-1):
            ideas[i]=ideas[i][0:len(ideas[i])-1]
    line_percentage=[0 for i in ideas]
    total_duration=0
    for i in range(len(ideas)):
        audio_file_path = os.path.join(audio_folder, f"{i}.mp3")
        if os.path.exists(audio_file_path):
            audio = AudioFileClip(audio_file_path)
            duration = audio.duration
            audio.close()
            total_duration += duration
            line_percentage[i]=duration
        else :
            return -i-1
    logo_path = 'logo.png'
    output_path = f"{id}video.mp4"
    final_video = f"{id}final.mp4"
    final_video_with_narration = f"{id}final_video.mp4"
    if total_duration/30>len(image_paths) :
        i=0
        n=len(image_paths)
        while total_duration/30>len(image_paths):
            image_paths.append(image_paths[i%n])
            i+=1
    Thread(target=phaseONE,args=(id,image_paths,output_path,total_duration,fps,final_video,
                                 final_video_with_narration,logo_path,line_percentage,ideas
                                 ,lang,text_color,text_bg_color,audio_folder)).start()
    return total_duration

def phaseONE(id,image_paths, output_path, total_duration, fps,final_video,
             final_video_with_narration,logo_path,line_percentage,ideas,lang,text_color,text_bg_color,audio_folder):
    create_transition_video(image_paths, output_path, total_duration, fps)

    phaseTWO(id,output_path,final_video,final_video_with_narration,fps,
             logo_path,line_percentage,ideas,lang,text_color,text_bg_color,audio_folder)

def phaseTWO(id,output_path,final_video,final_video_with_narration,
             fps,logo_path,line_percentage,ideas,lang,text_color,text_bg_color,audio_folder):
    update_video_phase(id,2)
    video_path = output_path
    output_path = f"{id}output.mp4"
    add_blurred_split_screen_effect(video_path,output_path)
    phaseTHREE(id,output_path,final_video,final_video_with_narration,fps,logo_path,
               line_percentage,ideas,lang,text_color,text_bg_color,audio_folder)

def phaseTHREE(id,output_path,final_video,final_video_with_narration,fps,logo_path,
               line_percentage,ideas,lang,text_color,text_bg_color,audio_folder):
    update_video_phase(id,3)
    cap = cv2.VideoCapture(output_path)
    start_frames = [int(sum(line_percentage[:i]) * fps) for i in range(len(line_percentage))]
    end_frames = [int(sum(line_percentage[:i+1]) * fps) for i in range(len(line_percentage))]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(final_video, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)
        for i, (start, end) in enumerate(zip(start_frames, end_frames)):
            if start <= frame_num < end:
                font_path = 'ReadexPro-Bold.ttf'
                font_size = 48
                font = ImageFont.truetype(font_path, font_size)
                text = ideas[i]
                textsize = font.getsize_multiline(text)

                if lang == 'ar':
                    textX = frame.shape[1] - 100 - textsize[0]
                else:
                    textX = 100
                textY = frame.shape[0] - 150

                padding = 10
                max_text_width = frame.shape[1] - 2 * (textX + padding)
                wrapped_lines = []
                words = text.split()
                line = ""
                for word in words:
                    test_line = line + word + " "
                    test_line_width = font.getsize_multiline(test_line)[0]
                    if test_line_width <= max_text_width:
                        line = test_line
                    else:
                        wrapped_lines.append(line)
                        line = word + " "
                wrapped_lines.append(line)
                wrapped_text_height = len(wrapped_lines) * textsize[1]
                wrapped_text_start_y = textY - wrapped_text_height + textsize[1] + padding
                for line in wrapped_lines:
                    text_width, text_height = font.getsize(line)
                    bg_left = textX - padding
                    bg_top = wrapped_text_start_y - padding
                    bg_right = textX + text_width + padding
                    bg_bottom = bg_top + text_height + padding

                    draw.rectangle((bg_left, bg_top, bg_right, bg_bottom), fill=text_bg_color)

                    text_position = (textX, wrapped_text_start_y)

                    draw.text(text_position, line, fill=text_color, font=font)

                    wrapped_text_start_y += textsize[1] + 2.5*padding
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)
        frame_num += 1
    cap.release()
    out.release()
    phaseFOUR(id,final_video,final_video_with_narration,logo_path,line_percentage,audio_folder)

def phaseFOUR(id,final_video,final_video_with_narration,logo_path,line_percentage,audio_folder):
    update_video_phase(id,4)
    video_clip = VideoFileClip(final_video)
    audio_clips = []
    for i in range(len(line_percentage)):
        audio_clip = AudioFileClip(os.path.join(audio_folder,f"{i}.mp3"))
        audio_clips.append(audio_clip)
    concatenated_audio = concatenate_audioclips(audio_clips)

    video_clip = video_clip.set_audio(concatenated_audio)

    video_clip.write_videofile(final_video_with_narration, codec="libx264", audio_codec="aac")
    for audio_clip in audio_clips:
        audio_clip.close()
    phaseFIVE(id,logo_path,line_percentage,final_video_with_narration,audio_folder)

def phaseFIVE(id,logo_path,line_percentage,final_video_with_narration,audio_folder):
    update_video_phase(id,5)
    video_clip = VideoFileClip(final_video_with_narration)

    logo = ImageClip(logo_path)
    logo = logo.resize(height=int(video_clip.h * 0.1))

    logo = logo.set_duration(video_clip.duration)

    logo_x_pos = 20
    logo_y_pos = "top"

    logo = logo.set_position((logo_x_pos, logo_y_pos))

    final_clip = CompositeVideoClip([video_clip, logo])

    final_clip.write_videofile(f"{id}.mp4", codec='libx264', audio_codec='aac')

    final_clip.close()
    video_clip.close()
    update_video_status(id,"Ready")
    removeUnwantedFiles(id,line_percentage,audio_folder)

def removeUnwantedFiles(id,line_percentage,audio_folder):
    os.remove(f'{id}final_video.mp4')
    os.remove(f'{id}final.mp4')
    os.remove(f'{id}video.mp4')
    os.remove(f'{id}output.mp4')
    for i in range(len(line_percentage)):
        os.remove(os.path.join(audio_folder, f'{i}.mp3'))
    print("all unwanted files are successfully removed!!!")



def zoom_in(image_path, output_path, duration=3, fps=30, max_scale=1.2):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    scale_increment = (max_scale - 1.0) / (duration * fps)

    center_x, center_y = width // 2, height // 2

    current_scale = 1.0
    for i in range(int(duration * fps)):
        current_scale += scale_increment

        translation_x = center_x * (1 - current_scale)
        translation_y = center_y * (1 - current_scale)

        transformation_matrix = np.float32([[current_scale, 0, translation_x], [0, current_scale, translation_y]])

        zoomed_in_image = cv2.warpAffine(image, transformation_matrix, (width, height))

        out.write(zoomed_in_image)

    out.release()


def zoom_out(image_path, output_path, duration=3, fps=30, max_scale=1.2):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    scale_decrement = (max_scale - 1.0) / (duration * fps)

    center_x, center_y = width // 2, height // 2

    current_scale = max_scale
    for i in range(int(duration * fps)):
        current_scale -= scale_decrement

        translation_x = center_x * (1 - current_scale)
        translation_y = center_y * (1 - current_scale)

        transformation_matrix = np.float32([[current_scale, 0, translation_x], [0, current_scale, translation_y]])

        zoomed_out_image = cv2.warpAffine(image, transformation_matrix, (width, height))

        out.write(zoomed_out_image)

    out.release()

def translate_left(image_path, output_path, duration=3, fps=30):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    visible_width = int(width * 0.8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (visible_width, height))

    total_translation_distance = width - visible_width

    translation_per_second = total_translation_distance / duration

    for i in range(int(duration * fps)):
        translation_distance = int(i / fps * translation_per_second)

        translation_matrix = np.float32([[1, 0, -translation_distance], [0, 1, 0]])

        translated_image = cv2.warpAffine(image, translation_matrix, (visible_width, height))

        out.write(translated_image)

    out.release()


def translate_right(image_path, output_path, duration=3, fps=30):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    visible_width = int(width * 0.8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (visible_width, height))

    total_translation_distance = width - visible_width

    translation_per_second = total_translation_distance / duration

    frames = []
    for i in range(int(duration * fps)):
        translation_distance = int(i / fps * translation_per_second)

        translation_matrix = np.float32([[1, 0, -translation_distance], [0, 1, 0]])

        translated_image = cv2.warpAffine(image, translation_matrix, (visible_width, height))

        frames.append(translated_image)

    for frame in reversed(frames):
        out.write(frame)

    out.release()

def concatenate_videos(video_paths, output_path, target_resolution=(1280,720)):
    video_clips = [VideoFileClip(video_path).resize(target_resolution) for video_path in video_paths]

    final_clip = concatenate_videoclips(video_clips, method="compose")

    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    for video_clip in video_clips:
        video_clip.close()


def create_transition_video(image_paths, output_path, total_duration, fps=30, animations=None):
    if not animations:
        animations = [zoom_in, translate_left, zoom_out, translate_right]

    video_paths = []
    i=0
    for image_path in image_paths:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(current_directory, f'temp_video_for_image_n_{i}.mp4')
        create_image_animation=random.choice(animations)
        create_image_animation(image_path, video_path, total_duration / len(image_paths), fps)
        video_paths.append(video_path)
        i+=1

    concatenate_videos(video_paths, output_path)

    for video_path in video_paths:
        os.remove(video_path)

def add_light_color_effect(arr, alpha=0.5, beta=0.5):
    yellow_tint = np.zeros_like(arr)
    yellow_tint[:, :, 0] = 0
    yellow_tint[:, :, 1] = 0
    yellow_tint[:, :, 2] = 3

    return cv2.addWeighted(arr, alpha, yellow_tint, beta, 0)

def gaussian_blur(arr, sigma):
    return cv2.GaussianBlur(arr, (0, 0), sigma)

def add_blurred_split_screen_effect(input_video_path, output_video_path, padding_percentage=0.05, blur_sigma=5):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    left_padding = int(width * padding_percentage)
    right_padding = int(width * padding_percentage)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left_blurred = frame[:, :left_padding].copy()
        left_blurred = gaussian_blur(left_blurred, sigma=blur_sigma)
        left_blurred = add_light_color_effect(left_blurred)

        right_blurred = frame[:, -right_padding:].copy()
        right_blurred = gaussian_blur(right_blurred, sigma=blur_sigma)
        right_blurred = add_light_color_effect(right_blurred)

        combined_frame = frame.copy()
        combined_frame[:, :left_padding] = left_blurred
        combined_frame[:, -right_padding:] = right_blurred

        out.write(combined_frame)

    cap.release()

    out.release()
    cv2.destroyAllWindows()