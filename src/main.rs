extern crate opencv;

use opencv::{
    core::{self, Size},
    highgui, imgcodecs,
    imgproc::{self, contour_area},
    prelude::VideoCaptureTrait,
    videoio::{self, VideoCaptureTraitConst},
};

#[derive(Debug, Clone)]
struct Finger {
    id: i32,
    history: Vec<(i32, i32)>,
    active: i32, // 0 newst, 4 oldest
}

// Detects fingers in an image for multitoch applications
fn main() {
    let project_root = project_root::get_project_root().expect("Failed to get project root");

    let image1_path = project_root.join("src").join("1.jpg");
    // let image2_path = project_root.join("src").join("2.jpg");
    let video_path = project_root.join("src").join("mt_camera_raw.AVI");

    let background =
        imgcodecs::imread(image1_path.to_str().expect("Invalid image1 path"), 1).unwrap();
    // let image: core::Mat = imgcodecs::imread(image2_path.to_str().expect("Invalid image2 path"), 1).unwrap();

    video(
        video_path.to_str().expect("Invalid video path"),
        &background,
    );
}

fn video(video_path: &str, background: &core::Mat) {
    let mut cap = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY).unwrap();

    let mut frame_counter = 0;

    let total_frames = cap
        .get(videoio::VideoCaptureProperties::CAP_PROP_FRAME_COUNT.into())
        .unwrap();
    let total_frames = total_frames as i32;

    let mut gray_background = background.clone();
    imgproc::cvt_color_def(&background, &mut gray_background, imgproc::COLOR_BGR2GRAY).unwrap();

    let mut fingers: Vec<Finger> = Vec::new();

    highgui::named_window("multi-touch", highgui::WINDOW_NORMAL).unwrap();
    highgui::resize_window("multi-touch", 720, 480).unwrap();

    loop {
        // if the last frame of the video is reached, reset the frame_counter and start again
        if frame_counter == total_frames {
            frame_counter = 0;
            cap.set(videoio::CAP_PROP_POS_FRAMES, 0.0).unwrap();
        }

        let mut frame = core::Mat::default();
        let success = cap.read(&mut frame).unwrap();

        if success {
            let mut image = frame.clone();

            // My code

            let temp_image = image.clone();
            imgproc::cvt_color_def(&temp_image, &mut image, imgproc::COLOR_BGR2GRAY).unwrap();

            let mut finger_coordinates: Vec<(i32, i32)> = Vec::new();
            image = prepare_image(&image, &gray_background);
            let ellipse_image = detect_fingers(&image, &frame, &mut finger_coordinates);
            // println!("\n \nFinger coordinates: {:?}", finger_coordinates);

            manage_fingers(&mut fingers, &finger_coordinates);

            image = ellipse_image;
            // End of my code

            highgui::imshow("multi-touch", &image).unwrap();
            let key = highgui::wait_key(50).unwrap();
            if key == 27 {
                // escape key
                break;
            }
            frame_counter += 1;
        }
    }
}

fn manage_fingers(fingers: &mut Vec<Finger>, finger_coordinates: &Vec<(i32, i32)>) {
    // No Existing fingers
    if fingers.is_empty() {
        for (i, finger) in finger_coordinates.iter().enumerate() {
            let mut new_finger = Finger {
                id: i as i32,
                history: Vec::new(),
                active: 0,
            };
            new_finger.history.push(*finger);
            fingers.push(new_finger);
        }
        return;
    }

    // Existing fingers
    for finger_coordinate in finger_coordinates {
        let nearest_finger = find_nearest_fingers(*finger_coordinate, fingers.clone(), 100);
        if nearest_finger.id == -1 {
            let mut new_finger = Finger {
                id: fingers.len() as i32,
                history: Vec::new(),
                active: 0,
            };
            new_finger.history.push(*finger_coordinate);
            fingers.push(new_finger);
        } else {
            let nearest_finger = fingers
                .iter_mut()
                .find(|finger| finger.id == nearest_finger.id)
                .unwrap();
            nearest_finger.history.push(*finger_coordinate);
        }
    }

    //Inactive fingers
    fingers.clone().iter_mut().for_each(|finger| {
        if finger.active < 4 {
            finger.active += 1;
        } else {
            fingers.retain(|f| f.id != finger.id);
        }
    });
}

// Detects the contours of the fingers in the image
fn detect_fingers(
    image: &core::Mat,
    original_frame: &core::Mat,
    finger_coordinates: &mut Vec<(i32, i32)>,
) -> core::Mat {
    let mut contours = core::Vector::<core::Vector<core::Point>>::new();
    let mut hierarchy = core::Vector::<core::Vec4i>::new();
    imgproc::find_contours_with_hierarchy_def(
        &image,
        &mut contours,
        &mut hierarchy,
        imgproc::RETR_CCOMP,
        imgproc::CHAIN_APPROX_SIMPLE,
    )
    .unwrap();

    let mut temp_original_frame = original_frame.clone();

    if hierarchy.is_empty() {
        return temp_original_frame;
    }

    let mut idx: i32 = 0;
    while idx >= 0 {
        let con = contours.get(idx.try_into().unwrap()).unwrap();
        idx = hierarchy.get(idx.try_into().unwrap()).unwrap()[0];

        if contour_area(&con, false).unwrap() > 30.00 && con.len() > 4 {
            let ellipse = imgproc::fit_ellipse_direct(&con).unwrap();

            let center_32 = core::Point::new(ellipse.center.x as i32, ellipse.center.y as i32);

            let major_axis = (ellipse.size.width / 2.0) as f64;
            let minor_axis = (ellipse.size.height / 2.0) as f64;

            let size_i32 = Size::from((major_axis as i32, minor_axis as i32));

            if major_axis > minor_axis * 2.5 || minor_axis > major_axis * 2.5 {
                // println!("Ellipse is too long");
                continue;
            }

            let ellipse_area = std::f64::consts::PI * (major_axis / 2.0) * (minor_axis / 2.0);
            if !(5.0..=150.0).contains(&ellipse_area) {
                // println!("Ellipse too big / small");
                continue;
            }

            finger_coordinates.push((center_32.x, center_32.y));

            imgproc::ellipse(
                &mut temp_original_frame,
                center_32,
                size_i32,
                ellipse.angle.into(),
                0.0,
                360.0,
                core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                1,
                8,
                0,
            )
            .unwrap();

            /*
            let _ = imgproc::draw_contours(
                &mut temp_original_frame,
                &contours,
                idx,
                core::Scalar::new(255.0, 0.0, 0.0, 0.0),
                1,
                8,
                &hierarchy,
                0,
                core::Point::new(0, 0),
            );
            */
        }
    }
    temp_original_frame
}

// Prepare image for finger contour detection
fn prepare_image(image: &core::Mat, background: &core::Mat) -> core::Mat {
    // subtract background from image
    let temp_image = image.clone();
    let mut img_subtraction = image.clone();
    //core::absdiff(&temp_image, &background, &mut img_subtraction).unwrap();
    core::subtract(
        &temp_image,
        &background,
        &mut img_subtraction,
        &core::no_array(),
        -1,
    )
    .unwrap();

    // first blur
    let temp_image = img_subtraction.clone();
    let mut img_blur = core::Mat::default();
    imgproc::blur(
        &temp_image,
        &mut img_blur,
        Size::from((20, 20)),
        Default::default(),
        0,
    )
    .unwrap();
    // subtract blur from subtraction
    let mut img_subtraction2 = core::Mat::default();

    //core::absdiff(&img_subtraction, &img_blur, &mut img_subtraction2).unwrap();

    core::subtract(
        &img_subtraction,
        &img_blur,
        &mut img_subtraction2,
        &core::no_array(),
        -1,
    )
    .unwrap();
    let img_blur2 = img_subtraction2.clone();
    /*
    // second blur
    img_blur2 = core::Mat::default();
    imgproc::blur(
        &img_subtraction2,
        &mut img_blur2,
        Size::from((1, 1)),
        Default::default(),
        0,
    )
    .unwrap();
    */
    // grayscale threshold
    let mut img_gray = core::Mat::default();
    imgproc::threshold(
        &img_blur2,
        &mut img_gray,
        12.0,
        255.0,
        imgproc::THRESH_BINARY,
    )
    .unwrap();
    // third blur
    let mut img_blur3 = core::Mat::default();
    imgproc::blur(
        &img_gray,
        &mut img_blur3,
        Size::from((5, 5)),
        Default::default(),
        0,
    )
    .unwrap();

    img_blur3
}

fn find_nearest_fingers(
    finger_coordinates: (i32, i32),
    fingers: Vec<Finger>,
    distance_to_finger_threshold: i32,
) -> Finger {
    let mut current_min_distance = 999999; // arbitrary large number
    let mut nearest_finger = Finger {
        id: -1,
        history: Vec::new(),
        active: -1,
    };

    for finger in fingers {
        let finger_location = finger.history.last().unwrap();

        let distance = (finger_coordinates.0 - finger_location.0).pow(2)
            + (finger_coordinates.1 - finger_location.1).pow(2);

        if distance < current_min_distance && distance < distance_to_finger_threshold {
            current_min_distance = distance;
            nearest_finger = finger.clone();
        }
    }
    nearest_finger
}
