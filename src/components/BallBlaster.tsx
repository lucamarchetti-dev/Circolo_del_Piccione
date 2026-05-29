import React, { useEffect, useRef, useState } from 'react';

interface ExtendedWindow extends Window {
    webkitAudioContext?: typeof AudioContext;
}

interface WristState {
    wristBaselineY: number;
    wristSmoothedY: number;
    wristNormalizedDelta: number;
}

interface PNG {
    active: boolean;
    x: number;
    y: number;
    width: number;
    height: number;
    color?: string;
}

interface Base { 
    canRestart: boolean;
    score: number;
    // gameSpeed: number; (prossime versioni)
    lastTime: number;
    pngPool: PNG[];
    objectSpawnTimer: number;
    animationId: number;
    visionAnimationId: number;
    object: ObjectEntity;
    hasStarted: boolean;
    cameraReady: boolean;
    // highScore: number; (prossime versioni)
}

// sezione high score (prossime versioni)

// #region Object
type ObjectState = 'ACTIVE' | 'INACTIVE';

interface ObjectEntity {
    x: number;
    y: number;
    width: number;
    height: number;
    speed: number;
    active: boolean;
    // draw: (ctx: CanvasRenderingContext2D) => void;
    reset: () => void;
    update: (dt: number, onStep?: () => void) => void;
    
}

const PNG_W = 40;
const PNG_H = 40;
// #endregion


const GameConfig = {
    VIDEO_WIDTH: 800,
    VIDEO_HEIGHT: 300,
    Object_Start: 400,
    COLORS: {
        PRIMARY: '#535353',
        ACCENT: '#ff5252',
        FOCUS: '#F59E0B',
        WHITE: '#ffffff',
    } // non so nemmeno se e in cosa è utile ma per il momento lo teniamo

}

const png = new Image();
png.src = "/assets/ball.png";

const spawnaPNG = (): ObjectEntity => ({
    x: GameConfig.Object_Start,
    width: PNG_W,
    height: PNG_H,
    active: true,

    reset() {
        this.active = true;
        this.width = PNG_W;
        this.height = PNG_H;
    },

    draw(ctx: CanvasRenderingContext2D) {
        const ix = Math.floor(this.x);
        const iy = Math.floor(this.y);

        if (!spriteReady) {
            ctx.fillStyle = GameConfig.COLORS.PRIMARY;
            ctx.fillRect(ix, iy, this.width, this.height);
            return;
        }

        let key: keyof typeof SPRITE_CONFIG.LDPI;
        switch (this.active) {
            case true:
                key = this.currentFrame === 0 ? 'TREX_WAIT_1' : 'TREX_WAIT_2';
                break;
            case false:
                key = this.currentFrame === 0 ? 'TREX_RUN_1' : 'TREX_RUN_2';
                break;
            default:
                key = 'TREX_RUN_1';
        }

        const src = SPRITE_FRAMES[key];
        const destW = this.ducking ? DINO_DUCKING_W : DINO_STANDING_W;
        const destH = this.ducking ? DINO_DUCKING_H : DINO_STANDING_H;
        const destY = this.ducking ? iy + (DINO_STANDING_H - DINO_DUCKING_H) : iy;

        ctx.drawImage(spriteImage, src.x, src.y, src.w, src.h, ix, destY, destW, destH);
    },