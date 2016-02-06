function rgb2hex(rgb){
    rgb = rgb.match(/^rgba?[\s+]?\([\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?/i);
    return (rgb && rgb.length === 4) ?
        ("0" + parseInt(rgb[1],10).toString(16)).slice(-2) +
        ("0" + parseInt(rgb[2],10).toString(16)).slice(-2) +
        ("0" + parseInt(rgb[3],10).toString(16)).slice(-2) : "";
}

var get_pixels = function() {
    var x = new Array(100);
    var frame = PIXEL.getCurrentFrame();

    for (var i=0;  i < 10;  i++) {
        for (var j=0;  j < 10;  j++) {
            var idx = i*10 + j;
            if (frame.data[i][j] == "rgba(0, 0, 0, 0)") {
                x[idx] = 1;
            } else {
                x[idx] = parseInt(frame.data[i][j].substring(1));
            }
        }
    }
    return x;
}

var clip_pixel = function(x){
    if(x>1) return 255;
    else if(x<-1) return 0;
    else return 255*(x+1.0)/2.0;
}

// mouse down event callback
function mouseDownCallback(e) {
    PIXEL.setDraw(true);
    var coordinates = getCoordinates(e);
    
    PIXEL.doAction(coordinates.x, coordinates.y, currentColor);
}

// mouse move event callback
function mouseMoveCallback(e) {
    var coordinates = getCoordinates(e);
    
    PIXEL.doAction(coordinates.x, coordinates.y, currentColor);
    e.preventDefault();
}

// mouse up event callback
function mouseUpCallback() {
    PIXEL.setDraw(false);
}

$(document).ready(function() {
    $('.slick').slick({
        slidesToShow: 2,
        autoplay: true,
        dots: true,
        autoplaySpeed: 3000,
        responsive: [
            {
                breakpoint: 480,
                settings: {
                    slidesToShow: 1,
                    slidesToScroll: 1
                }
            }
        ]
    });
    $('.turing-slick').slick({
        slidesToShow: 6,
        autoplay: true,
        dots: true,
        autoplaySpeed: 3000,
        responsive: [
            {
                breakpoint: 480,
                settings: {
                    slidesToShow: 3,
                    slidesToScroll: 3
                }
            }
        ]
    });

    $("[data-toggle=tooltip]").tooltip();

    var layer_defs = [];
    layer_defs.push({type:"input", out_sx:1, out_sy:1, out_depth:100});
    layer_defs.push({type:"deconv", sx:4, filters:512, stride:1, pad:0, bn:true, activation:"relu"});
    layer_defs.push({type:"deconv", sx:4, filters:256, stride:2, pad:1, bn:true, activation:"relu"});
    layer_defs.push({type:"deconv", sx:4, filters:128, stride:2, pad:1, bn:true, activation:"relu"});
    layer_defs.push({type:"deconv", sx:4, filters:64, stride:2, pad:1, bn:true, activation:"relu"});
    layer_defs.push({type:"deconv", sx:4, filters:3, stride:2, pad:1, activation:"tanh"});

    var net = new convnetjs.Net();
    net.makeLayers(layer_defs);

    net.layers[1].fromJSON(layer_0);
    net.layers[3].fromJSON(layer_1);
    net.layers[5].fromJSON(layer_2);
    net.layers[7].fromJSON(layer_3);
    net.layers[9].fromJSON(layer_4);

    var input = new convnetjs.Vol(1, 1, 100, 0.0);

    var draw = function() {
        var output = net.forward(input);
        var scale = 2;
        var W = output.sx * scale;
        var H = output.sy * scale;

        var canv = document.createElement("canvas");
        canv.width = W;
        canv.height = H;

        var ctx = canv.getContext("2d");
        var g = ctx.createImageData(W, H);

        for(var d=0; d < 3; d++) {
            for(var x=0; x < output.sx; x++) {
                for(var y=0; y < output.sy; y++) {
                    var dval = clip_pixel(output.get(x,y,d));

                    for(var dx = 0; dx < scale; dx++) {
                        for(var dy =0 ;dy < scale; dy++) {
                            var pp = ((W * (y*scale + dy)) + (dx + x*scale)) * 4;
                            g.data[pp + d] = dval;
                            if(d===0) g.data[pp+3] = 255; // alpha channel
                        }
                    }
                }
            }
        }
        ctx.putImageData(g, 0, 0);

        document.getElementById("images").appendChild(canv);

        $(canv).attr('data-original-title', canvas[0].outerHTML)
               .tooltip({html:true})
               .hide()
               .fadeIn(1000);
    }

    $("#fakeLoader").fadeOut(3000);

    $("#draw").click(draw);
    $("#shuffle").click(draw);
});


// deactivate element
function deactivate($el) {
    return $el.removeClass("active");
}

// activate element
function activate($el) {
    return $el.addClass("active");
}

// disable element
function disable($el) {
    return $el.addClass("disabled");
}

// enable element
function enable($el) {
    return $el.removeClass("disabled");
}

// is element enabled?
function isEnabled($el) {
    return $el.size() > 0 && !$el.hasClass("disabled");
}

// track event
function trackEvent(e, url) {
    _gaq.push(['_trackEvent', 'Drawings', e, url]);
}

// returns mouse or tap event relative coordinates
function getCoordinates(e) {
    var x, y;
    
    x = e.offsetX ? e.offsetX : e.pageX - e.target.parentNode.offsetLeft;
    y = e.offsetY ? e.offsetY : e.pageY - e.target.parentNode.offsetTop;
    
    return {x: x, y: y};
}

var currentColor = "#000000",
    copyFrameIndex = -1,
    tips = true;

// mouse down event callback
function mouseDownCallback(e) {
    PIXEL.setDraw(true);
    var coordinates = getCoordinates(e);
    
    PIXEL.doAction(coordinates.x, coordinates.y, currentColor);
}

// mouse move event callback
function mouseMoveCallback(e) {
    var coordinates = getCoordinates(e);
    
    PIXEL.doAction(coordinates.x, coordinates.y, currentColor);
    e.preventDefault();
}

// mouse up event callback
function mouseUpCallback() {
    PIXEL.setDraw(false);
}

var canvas = $("#pixel");
PIXEL.init(canvas[0], true);

// set drawing on mousedown
canvas.mousedown(mouseDownCallback).mousemove(mouseMoveCallback);
canvas.bind('touchstart', mouseDownCallback).bind('touchmove', mouseMoveCallback);

// reset drawing on mouseup
$(document).mouseup(mouseUpCallback);
$(document).bind('touchend', mouseUpCallback);

// controls
$("#clear").click(function() {
    if(confirm("Sure?")) {
        clearAll();
    }
});

function clearAll() {
    var framesLength = PIXEL.getFramesLength();
    for(var i = framesLength-1; i >= 0; i--) {
        PIXEL.log('REMOVE ' + i + ' FRAME OF ' + PIXEL.getFramesLength() + ' FRAMES!');
        PIXEL.clearCanvasAt(i);
        PIXEL.removeFrame(i);
        disable($(".frame[data-frame=" + i + "]"));
    }
    PIXEL.setCurrentFrame(0);

    deactivate($(".frame.active"));
    activate($(".frame[data-frame=0]"));
    enable($(".frame[data-frame=0]"));
    disable($(".remove_frame"));
    enable($(".add_frame"));
    
    copyFrameIndex = -1;
}

$(".action.selectable").click(function() {
    PIXEL.setAction($(this).data('action'));
    
    deactivate($(".action.selectable.active"));
    activate($(this));
});

// colors
$(".color").click(function() {
    currentColor = $(this).data('color');
    
    deactivate($(".color.active"));
    activate($(this));
});

// undo
$(".undo").click(function() {
    PIXEL.undo();
});

// copy
$(".copy").click(function() {
    copyFrameIndex = PIXEL.getCurrentFrameId();
});

$(".paste").click(function() {
    if(copyFrameIndex > -1 && copyFrameIndex < PIXEL.getFramesLength()) {
        PIXEL.pasteFrameAt(copyFrameIndex);
    }
});

$(".rotate").click(function() {
    PIXEL.rotate();
});



// jQuery to collapse the navbar on scroll
function collapseNavbar() {
    if ($(".navbar").offset().top > 50) {
        $(".navbar-fixed-top").addClass("top-nav-collapse");
    } else {
        $(".navbar-fixed-top").removeClass("top-nav-collapse");
    }
}

$(window).scroll(collapseNavbar);
$(document).ready(collapseNavbar);

// jQuery for page scrolling feature - requires jQuery Easing plugin
$(function() {
    $('a.page-scroll').bind('click', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top
        }, 1500, 'easeInOutExpo');
        event.preventDefault();
    });
});

// Closes the Responsive Menu on Menu Item Click
$('.navbar-collapse ul li a').click(function() {
  if ($(this).attr('class') != 'dropdown-toggle active' && $(this).attr('class') != 'dropdown-toggle') {
    $('.navbar-toggle:visible').click();
  }
});

