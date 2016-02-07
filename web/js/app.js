function rgb2hex(rgb){
    rgb = rgb.match(/^rgba?[\s+]?\([\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?/i);
    return (rgb && rgb.length === 4) ?
        ("0" + parseInt(rgb[1],10).toString(16)).slice(-2) +
        ("0" + parseInt(rgb[2],10).toString(16)).slice(-2) +
        ("0" + parseInt(rgb[3],10).toString(16)).slice(-2) : "";
}

var make_z = function(max_length, scale) {
    var z = []
    var scale = scale | 2;
    while(z.length < max_length){
      var randomnumber=Math.random() * scale;
      var found=false;
      for(var i=0;i<z.length;i++){
        if(z[i]==randomnumber){found=true;break}
      }
      if(!found)z[z.length]=randomnumber;
    }
    return z;
}

var get_pixels = function() {
    var x = new Array(100);
    var frame = PIXEL.getCurrentFrame();

    for (var i=0;  i < 10;  i++) {
        for (var j=0;  j < 10;  j++) {
            var idx = i*10 + j;
            if (frame[i][j] == "rgba(0, 0, 0, 0)") {
                x[idx] = 0;
            } else {
                x[idx] = ((parseInt(frame[i][j].substring(1,3) ,16)/255*2)-1)/2;
            }
        }
    }
    return x;
}

var recover_pixels = function(x) {
    for (var i=0;  i < 10;  i++) {
        for (var j=0;  j < 10;  j++) {
            var idx = i*10 + j;
            x[idx] = ((x[idx] * 2) + 1) / 2 * 255;
        }
    }
    return x;
}

var draw_pixels = function(x) {
    var frame = PIXEL.getCurrentFrame();

    for (var i=0;  i < 10;  i++) {
        for (var j=0;  j < 10;  j++) {
            var hex = Math.ceil(x[i*10+j]).toString(16);
            PIXEL.setDraw(true);
            PIXEL.doAction(i*20, j*20, "#" + hex + hex + hex);
            PIXEL.setDraw(false);
        }
    }
}

var clip_pixel = function(x){
    if(x>1) return 255;
    else if(x<-1) return 0;
    else return 255*(x+1.0)/2.0;
}

function cloneCanvas(oldCanvas) {

    //create a new canvas
    var newCanvas = document.createElement('canvas');
    var context = newCanvas.getContext('2d');

    //set dimensions
    newCanvas.width = oldCanvas.width;
    newCanvas.height = oldCanvas.height;

    //apply the old canvas to the new one
    context.drawImage(oldCanvas, 0, 0);

    //return the new canvas
    return newCanvas;
}

$(document).ready(function() {
    draw_pixels(make_z(100, 255));

    $('.slick').slick({
        slidesToShow: 2,
        autoplay: true,
        dots: true,
        autoplaySpeed: 3000,
        responsive: [
            {
                breakpoint: 980,
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
                breakpoint: 1200,
                settings: {
                    slidesToShow: 5,
                    slidesToScroll: 5
                }
            },
            {
                breakpoint: 980,
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

    var duplicates = [];
    var pixels = [];

    var draw = function() {
        cur_pixel = get_pixels();
        input.w = cur_pixel;

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

        duplicates.push(cloneCanvas(document.getElementById("pixel")));
        pixels.push(cur_pixel);

        $(canv).tooltip({
                html: true,
                template: '<div class="tooltip"><div class="tooltip-inner pixel-tooltip"></div></div>',
                title: function(e) {
                     var duplicated = duplicates[parseInt($(this).attr("id")) - 1];
                     return duplicated;
                },
                }).hide()
                .attr("id", duplicates.length)
                .fadeIn(1000)
                .click(function() {
                    draw_pixels(recover_pixels(pixels[parseInt($(this).attr("id")) - 1]));
                });
    }

    $("#fakeLoader").fadeOut(3000);

    $("#draw").click(draw);
    $("#shuffle").click(function() {
        draw_pixels(make_z(100, 255));
        draw();
    });
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

