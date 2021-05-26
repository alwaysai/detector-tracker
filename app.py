import time
import edgeiq
"""
Use object detection and tracking to follow objects as they move across
the frame. Detectors are resource expensive, so this combination
reduces stress on the system, increasing the resulting bounding box output
rate. The detector is set to execute every 30 frames, but this can be
adjusted by changing the value of the `detect_period` variable.

To change the computer vision model, the engine and accelerator,
and add additional dependencies read this guide:
https://alwaysai.co/docs/application_development/configuration_and_packaging.html
"""


def object_enters(object_id, prediction):
    print("{}: {} enters".format(object_id, prediction.label))


def object_exits(object_id, prediction):
    print("{} exits".format(prediction.label, object_id))


def main():
    # The current frame index
    frame_idx = 0
    # The number of frames to skip before running detector
    detect_period = 30

    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    tracker = edgeiq.CorrelationTracker(
            max_objects=5,
            deregister_frames=detect_period+20,
            enter_cb=object_enters,
            exit_cb=object_exits)
    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            while True:
                frame = video_stream.read()
                predictions = []
                detect = frame_idx % detect_period == 0

                if detect:
                    results = obj_detect.detect_objects(
                            frame, confidence_level=.5)
                    predictions = results.predictions

                # Generate text to display on streamer
                text = ["Model: {}".format(obj_detect.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(
                            results.duration))
                text.append("Objects:")

                objects = tracker.update(predictions, frame)

                # Update the label to reflect the object ID
                tracked_predictions = []
                for (object_id, prediction) in objects.items():
                    # Use the original class label instead of the prediction
                    # label to avoid iteratively adding the ID to the label
                    class_label = obj_detect.labels[prediction.index]
                    prediction.label = "{}: {}".format(object_id, class_label)
                    text.append("{}".format(prediction.label))
                    tracked_predictions.append(prediction)

                frame = edgeiq.markup_image(
                        frame, tracked_predictions, show_labels=True,
                        show_confidences=False, colors=obj_detect.colors)
                streamer.send_data(frame, text)
                frame_idx += 1
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
