package com.seeta.sdk;

public class FaceInfo
{
	/* rectangle bounding box */
	public int x;
	public int y;
	public int width;
	public int height;

	/* view of face (Euler angle) */
	public double roll;  /**< rotation around x aixs */
	public double pitch; /**< rotation around y axis */
	public double yaw;   /**< rotation around z axis */

	public double score;
}
