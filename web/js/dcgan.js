$(document).ready(function() {
  var maxmin = function(w) {
    if(w.length === 0) { return {}; } // ... ;s

    var maxv = w[0];
    var minv = w[0];
    var maxi = 0;
    var mini = 0;
    for(var i=1;i<w.length;i++) {
      if(w[i] > maxv) { maxv = w[i]; maxi = i; } 
      if(w[i] < minv) { minv = w[i]; mini = i; } 
    }
    return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv:maxv-minv};
  }

  var layer_defs, net;
  layer_defs = [];
  layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:100});
  layer_defs.push({type:'deconv', sx:4, filters:512, stride:1, pad:0, bn:true, activation:'relu'});
  layer_defs.push({type:'deconv', sx:4, filters:256, stride:2, pad:1, bn:true, activation:'relu'});
  layer_defs.push({type:'deconv', sx:4, filters:128, stride:2, pad:1, bn:true, activation:'relu'});
  layer_defs.push({type:'deconv', sx:4, filters:64, stride:2, pad:1, bn:true, activation:'relu'});
  layer_defs.push({type:'deconv', sx:4, filters:3, stride:2, pad:1, activation:'tanh'});

  net = new convnetjs.Net();
  net.makeLayers(layer_defs);

  net.layers[1].fromJSON(layer_0);
  net.layers[3].fromJSON(layer_1);
  net.layers[5].fromJSON(layer_2);
  net.layers[7].fromJSON(layer_3);
  net.layers[9].fromJSON(layer_4);

  var sliders = [];
  var strength = []
  var z0 = [];

  var num_tags = 123;

  var initialize = function(){
    var doms = [];
    dom = document.createElement('div');
    //dom.className = "box";
    dom.innerHTML = "noise_strength" + "<br>";
    doms.push(dom);
    document.getElementById('z').appendChild(doms[0]);
    strength.push(document.createElement('input'));
    strength[0].type="range";
    strength[0].value=100;
    strength[0].min='0';
    strength[0].max='100';
    strength[0].step='1';
    doms[0].appendChild(strength[0]);
    for(var i=0; i<num_tags; i++){
      dom = document.createElement('input');
      dom.className = "slider";
      dom.innerHTML = zws['w'][i]['name'] + "<br>";
      doms.push(dom);
      document.getElementById('z').appendChild(doms[i]);
    }
    $('.slider').slider({
      formatter: function(value) {
        return 'Current value: ' + value;
      }
    });
    for(var i=0; i<num_tags; i++){
      sliders.push(document.createElement('input'));
      sliders[i].type="range";
      sliders[i].value=0;
      sliders[i].min='-100';
      sliders[i].max='100';
      sliders[i].step='1';
      doms[i+1].appendChild(sliders[i]);
    }
    for(var i=0; i<100; i++){
      //z0.push(Math.random()*2.0-1.0);
      z0.push(0);
    }
    test();
  }

  $("#draw").click(function(){
    test();
  });

  $("#shuffle").click(function(){
    for(var i=0; i<100; i++){
      //sliders[i].value=Math.random()*100;
      z0[i] = (Math.random()*2.0-1.0);
    }
    test();
  });

  var reset_value = function(){
    for(var i=0; i<num_tags; i++){
      sliders[i].value=0;
    }
  }

  var test = function(){
    x = new convnetjs.Vol(1,1,100,0.0);
    for(var i=0; i<100; i++){
      var z = strength[0].value/100.0*z0[i];
      for(var j=0; j<num_tags; j++){
        z += sliders[j].value/200.0 * zws['w'][j]['v'][i];
      }
      if(z>1) z=1;
      else if(z<-1) z=-1;
      x.set(0,0,i,z);
    }
    img = net.forward(x);
    draw_activations_COLOR(img);
  }


  var clip_pixel = function(x){
    if(x>1) return 255;
    else if(x<-1) return 0;
    else return 255*(x+1.0)/2.0;
  }

  var draw_activations_COLOR = function(A, scale, grads) {
    var s = scale || 2; // scale
    var draw_grads = false;
    if(typeof(grads) !== 'undefined') draw_grads = grads;

    var w = draw_grads ? A.dw : A.w;
    var mm = maxmin(w);

    var canv = document.createElement('canvas');
    canv.className = 'actmap';
    var W = A.sx * s;
    var H = A.sy * s;
    canv.width = W;
    canv.height = H;
    var ctx = canv.getContext('2d');
    var g = ctx.createImageData(W, H);
    for(var d=0;d<3;d++) {
      for(var x=0;x<A.sx;x++) {
        for(var y=0;y<A.sy;y++) {
          var dval = clip_pixel(A.get(x,y,d));
          for(var dx=0;dx<s;dx++) {
            for(var dy=0;dy<s;dy++) {
              var pp = ((W * (y*s+dy)) + (dx + x*s)) * 4;
              g.data[pp + d] = dval;
              if(d===0) g.data[pp+3] = 255; // alpha channel
            }
          }
        }
      }
    }
    ctx.putImageData(g, 0, 0);
    document.getElementById('hoge').appendChild(canv);
    $(canv).hide().fadeIn(1000);
  }

  initialize();
}); 