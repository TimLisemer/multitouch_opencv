extern crate opencv;

use opencv::core::Size;
use opencv::{core, highgui, imgcodecs, imgproc};

fn main() {
    let project_root = project_root::get_project_root().expect("Failed to get project root");

    let image1_path = project_root.join("src").join("1.jpg");
    let image2_path = project_root.join("src").join("2.jpg");

    let background =
        imgcodecs::imread(image1_path.to_str().expect("Invalid image1 path"), 1).unwrap();
    let mut image =
        imgcodecs::imread(image2_path.to_str().expect("Invalid image2 path"), 1).unwrap();

    image = prepare_image(&image, &background);

    highgui::imshow("result", &image).unwrap();
    highgui::wait_key(0).unwrap();
    highgui::destroy_all_windows().unwrap();
}

fn prepare_image(image: &core::Mat, background: &core::Mat) -> core::Mat {
    // subtract background from image
    let temp_image = image.clone();
    let mut img_subtraction = image.clone();
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
    core::subtract(
        &img_subtraction,
        &img_blur,
        &mut img_subtraction2,
        &core::no_array(),
        -1,
    )
    .unwrap();
    // second blur
    let mut img_blur2 = core::Mat::default();
    imgproc::blur(
        &img_subtraction2,
        &mut img_blur2,
        Size::from((10, 10)),
        Default::default(),
        0,
    )
    .unwrap();
    // grayscale threshold
    let mut img_gray = core::Mat::default();
    imgproc::threshold(
        &img_blur2,
        &mut img_gray,
        15.0,
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
