from mmcv.transforms import Compose
from mmdet.apis import DetInferencer, inference_detector, init_detector
from mmdet.registry import VISUALIZERS
import PIL
import streamlit as st
import configs
import cv2
import numpy as np


def main():
    # Setting page layout
    st.set_page_config(
        page_title="Object Detection with Open-mmlab MMDetection",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Object Detection with Open-mmlab MMDetection")  # Main page heading
    st.sidebar.header("List of Available Tasks")  # Sidebar
    source_radio = st.sidebar.radio("Select Source", configs.SOURCES_LIST)

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['Object Detection'])

    default_imgs = {'Object Detection': configs.DEFAULT_IMAGE_MMDETECTION}

    default_predicted_imgs = {'Object Detection': configs.DEFAULT_DETECT_IMAGE}


    source_img = None
    # If image is selected
    if source_radio == configs.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)
        inferencer = DetInferencer(configs.detection_model_name, device=configs.device)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(default_imgs[model_type])
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image, caption="Default Image", use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    uploaded_image = np.array(uploaded_image)
                    st.image(source_img, caption="Uploaded Image", use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(default_predicted_imgs[model_type])
                st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
            else:
                if st.sidebar.button('Predict'):
                    try:
                        res = inferencer(uploaded_image, out_dir='./output')
                        annotated_image = res['visualization']
                        st.image(annotated_image, caption='Detected Image',
                                 use_column_width=True)

                    except Exception as ex:
                        st.write("No image is uploaded yet!")

    elif source_radio == configs.VIDEO:

        source_vid = st.sidebar.selectbox(
            "Choose a video...", configs.VIDEOS_DICT.keys())
        col1, col2 = st.columns(2)

        inferencer = init_detector(configs.config_path, configs.checkpoint, device=configs.device)
        inferencer.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        test_pipeline = Compose(inferencer.cfg.test_dataloader.dataset.pipeline)

        visualizer = VISUALIZERS.build(inferencer.cfg.visualizer)
        visualizer.dataset_meta = inferencer.dataset_meta

        with col1:
            video_filename = configs.VIDEOS_DICT.get(source_vid)
            with open(video_filename, 'rb') as video_file:
                video_bytes = video_file.read()
            # if video_bytes:
                st.video(video_bytes)
        with col2:
            if st.sidebar.button('Predict'):
                try:
                    video_reader = cv2.VideoCapture(str(video_filename))
                    st_frame = st.empty()
                    while video_reader.isOpened():
                        success, frame = video_reader.read()
                        if success:
                            result = inference_detector(inferencer, frame, test_pipeline=test_pipeline)
                            visualizer.add_datasample(
                                name='video',
                                image=frame,
                                data_sample=result,
                                draw_gt=False,
                                show=False,
                                pred_score_thr=0.3)
                            frame = visualizer.get_image()
                            st_frame.image(frame)
                        else:
                            video_reader.release()
                            break
                except Exception as e:
                    st.sidebar.error("Error loading video: " + str(e))
    else:
        st.error("Please select a valid source type!")


if __name__ == '__main__':
    main()