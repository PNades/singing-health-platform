// AudioWorklet capture (spec 5.1): mono float samples at the context's native rate.
// Posts {samples: Float32Array, rms: number} every BLOCK frames, always; the page
// decides whether to keep them (recording) or only show the level (meter).
const BLOCK = 1024;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(BLOCK);
    this.n = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const chans = input.length;
      const len = input[0].length;
      for (let i = 0; i < len; i++) {
        let v = 0;
        for (let c = 0; c < chans; c++) v += input[c][i];
        this.buf[this.n++] = v / chans;
        if (this.n === BLOCK) {
          let ss = 0;
          for (let k = 0; k < BLOCK; k++) ss += this.buf[k] * this.buf[k];
          this.port.postMessage({ samples: this.buf, rms: Math.sqrt(ss / BLOCK) }, [this.buf.buffer]);
          this.buf = new Float32Array(BLOCK);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
