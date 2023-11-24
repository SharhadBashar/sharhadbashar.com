let slideIndex = 1;
showSlides(slideIndex);

// Next/previous controls
function plusSlides(n) {
    showSlides(slideIndex += n);
}

// Thumbnail image controls
function currentSlide(n) {
    showSlides(slideIndex = n);
}

function showSlides(n) {
    let i;
    let slides = document.getElementsByClassName("mySlides");
    let dots = document.getElementsByClassName("demo");
    if (n > slides.length) {slideIndex = 1}
    if (n < 1) {slideIndex = slides.length}
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }
    for (i = 0; i < dots.length; i++) {
        dots[i].className = dots[i].className.replace(" active", "");
    }
    slides[slideIndex-1].style.display = "block";
    dots[slideIndex-1].className += " active";
}


function changeNavActive(nav_item) {
    alignToTop = true;
    var active = document.getElementsByClassName("nav-bar_item_active");
    active[0].className = active[0].className.replace(" nav-bar_item_active", "");
    document.getElementsByClassName(nav_item)[0].className += " nav-bar_item_active";
    document.getElementsByClassName(nav_item + "_container")[0].scrollIntoView();
}



function showContact() {
    var x = document.getElementById("contactus");
    if (x.style.display === "none") {
      x.style.display = "block";
    } else {
      x.style.display = "none";
    }
  }