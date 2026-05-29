import type { Keypoint, Pose } from '@tensorflow-models/pose-detection/dist/types';

export function getKeypoint(
    pose: Pose,
    name: string,
    minConfidence: number = 0.3
) : Keypoint | null{
    const keypoint = pose.keypoints.find(k => k.name === name);

    if(!keypoint)
        return null

    if(keypoint.score !== undefined && keypoint.score < minConfidence)
        return null

    return keypoint;
}

function drawPoint(
    ctx: CanvasRenderingContext2D,
    y: number,
    x: number,
    r: number = 5,
    color: string = 'aqua'
) : void {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
}

function drawLine(
    ctx: CanvasRenderingContext2D,
    a: { x: number, y: number },
    b: { x: number, y: number },
    color: string = 'lime',
    lineWidth: number = 2) : void {

    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
};

export function drawKeypoints(
    ctx: CanvasRenderingContext2D,
    pose: Pose,
    minConfidence: number = 0.3
) : void {
    for(const keypoint of pose.keypoints) {
        if(keypoint.score !== undefined && keypoint.score < minConfidence)
            continue;

        drawPoint(ctx, keypoint.y, keypoint.x);        
    }
}

export function drawArm(
    ctx: CanvasRenderingContext2D,
    pose: Pose,
    side: 'left' | 'right',
    minConfidence: number = 0.3
) : void {
    const shoulder = getKeypoint(pose, `${side}_shoulder`, minConfidence);
    const elbow = getKeypoint(pose, `${side}_elbow`, minConfidence);
    const wrist = getKeypoint(pose, `${side}_wrist`, minConfidence);

    if(!shoulder || !elbow || !wrist)
        return;

    drawLine(ctx, shoulder, elbow);
    drawLine(ctx, elbow, wrist);
}