image_ggplot <- function(image, interpolate = FALSE) {
  info <- image_info(image)
  ggplot2::ggplot(data.frame(x = 0, y = 0), ggplot2::aes_string('x', 'y')) +
    ggplot2::geom_blank() +
    ggplot2::theme_void() +
    ggplot2::scale_y_reverse() +
    ggplot2::coord_fixed(expand = FALSE, xlim = c(0, info$width), ylim = c(0, info$height)) +
    ggplot2::annotation_raster(image, 0, info$width, -info$height, 0, interpolate = interpolate) +
    NULL
}