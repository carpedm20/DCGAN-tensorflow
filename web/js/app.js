
window.mobilecheck = function() {
  var check = false;
  (function(a){if(/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0,4)))check = true})(navigator.userAgent||navigator.vendor||window.opera);
  return check;
}

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

$( document ).ready(function() {
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

    $(".draw").click(draw);
    $(".shuffle").click(function() {
        draw_pixels(make_z(100, 255));
        draw();
    });

    $("#loading").hide();
    $("#draw-btn").show();

    if (!mobilecheck()) {
        draw();
    }

    $("#fakeLoader").fadeOut(3000);
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

