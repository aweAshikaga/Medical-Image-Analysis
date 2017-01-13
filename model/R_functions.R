library("RGtk2")
library("EBImage")

adjustContrast <- function(img, factor) {
  return(img * factor)
}

adjustBrightness <- function(img, factor) {
  return(img+factor)
}

adjustGamma <- function(img, factor) {
  return(img^factor)
}

highPassFilter <- function(img) {
  fHigh <- matrix(1, ncol=3, nrow=3)
  fHigh[2,2] <- -8
  return(filter2(img, fHigh))
}

image <- gtkImage(filename = "C:/Users/Hendrik/Documents/Projektarbeit_R/Contrast\testImage.jpg")
imagePB <- gdkPixbufNewFromFile(filename = "C:/Users/Hendrik/Documents/Projektarbeit_R/Contrast/testImage.jpg")
imageEB <- readImage("C:/Users/Hendrik/Documents/Projektarbeit_R/Contrast/testImage.jpg")

#window <- gtkWindow()
#window$title <- "Contrast"
#frame <- gtkFrameNew("testFrame")
#window$add(image)


display(imageEB)
display(adjustContrast(imageEB,3))
display(adjustGamma(imageEB,5))
display(highPassFilter(imageEB))
