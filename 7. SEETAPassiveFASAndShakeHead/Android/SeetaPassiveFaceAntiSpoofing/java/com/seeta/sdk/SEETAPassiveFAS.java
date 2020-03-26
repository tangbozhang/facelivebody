package com.seeta.sdk;

import android.util.Log;

public class SEETAPassiveFAS
{
	static {
		System.loadLibrary("FaceSpoofingDetectJni");
	}

	public enum SystemState
	{
		noFace,
		detecting,
		pleaseTurn,
		passed,
		notPass,
		pleaseBlink,
		pleaseNodHead,
		pleaseFaceTheCamera
	}
	public long impl = 0;	// native object pointer
								   
	public SEETAPassiveFAS(String modelPath, int shakeHeadAngleThreshold, int nodHeadAngleThreshold,
						   double clarityThreshold, double fuseThreshold, SystemState systemState, int firstPhaseFrameNum,
								   int detectFrameNum)
	{
			this.construct(modelPath, shakeHeadAngleThreshold, nodHeadAngleThreshold,clarityThreshold,
						   fuseThreshold, systemState.ordinal(), firstPhaseFrameNum,
						   detectFrameNum);
	}
	
	public SystemState Detect(ImageData image, FaceInfo face, Point[] points)
	{
		int systemStateIndex = this.DetectCore(image, face , points);
		SystemState state = SystemState.values()[systemStateIndex];

		return state;
	}

	public void Reset(SystemState systemState){this.ResetCore(systemState.ordinal());}

	public  boolean SetActions(SystemState[] actions)
	{
		int length = actions.length;
		if(length < 0) return false;
		int[] actionIndexs = new int[length];
		int i = 0;
		for(;i < length; ++i)
		{
			actionIndexs[i] = actions[i].ordinal();
		}

		return this.SetActionsCore(actionIndexs);
	}

	public native void getLog(double[] valueArray);

	private native void construct(String modelPath, int shakeHeadAngleThreshold, int nodheadAngleThreshold,
								  double clarityThreshold, double fuseThreshold, int systemStateIndex, int firstPhaseFrameNum,
								  int detectFrameNum);
	private native int DetectCore(ImageData image, FaceInfo face, Point[] points);
	private native void ResetCore(int systemStateIndex);
	private native boolean SetActionsCore(int[] actionIndexs);

	protected void finalize() throws Throwable {
		super.finalize();
		this.dispose();
	}
	public native void dispose();
	
	public native ImageData getSnapshot();
}
