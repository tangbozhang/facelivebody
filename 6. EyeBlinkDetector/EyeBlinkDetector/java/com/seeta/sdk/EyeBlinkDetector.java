package com.seeta.sdk;

import android.util.Log;

public class EyeBlinkDetector
{
    static {
        Log.e("EyeBlinkDetector", "Start Load");
        System.loadLibrary("EyeBlinkDetectorJni");
        Log.e("EyeBlinkDetector", "Finish Load");
    }

	public long impl = 0;	// native object pointer

	private native void construct(String model);
	private native void construct(String model, String device);
	public native void dispose();

    public EyeBlinkDetector() {
        this("");
    }

	public EyeBlinkDetector(String model) {
		this.construct(model);
	}

	public EyeBlinkDetector(String model, String device) {
		this.construct(model, device);
	}

	protected void finalize() throws java.lang.Throwable {
        super.finalize();
		this.dispose();
    }
	
	public static int LEFT_EYE  =  1;
	public static int RIGHT_EYE  =  2;
	
	public native int Detect(ImageData image, Point[] landmarks);//返回二进制数表示是否眨眼，左眼闭上会把LEFT_EYE位置为1，右眼闭上会把RIGHT_EYE位置为1，
}
