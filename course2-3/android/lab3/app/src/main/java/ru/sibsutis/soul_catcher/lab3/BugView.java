package ru.sibsutis.soul_catcher.lab3;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.view.MotionEvent;
import android.view.View;

public class BugView extends View {

    private Bitmap background;
    private Paint score;

    BugsController bugsController;

    public BugView(Context context) {
        super(context);
        this.bugsController = new BugsController(5, this);

        background = BitmapFactory.decodeResource(context.getResources(), R.drawable.desk);
        score = new Paint();
        score.setColor(Color.BLACK);
        score.setTextAlign(Paint.Align.CENTER);
        score.setTextSize(75);
        score.setTypeface(Typeface.DEFAULT_BOLD);
        score.setAntiAlias(true);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        bugsController.update();
        int sc = bugsController.points;
        canvas.drawBitmap(background, 0, 0, null);
        canvas.drawText("score: " + sc, getWidth() / (float) 2, 50, score);

        bugsController.drawBugs(canvas);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            float eventX = event.getX();
            float eventY = event.getY();
            bugsController.touchEvent(eventX, eventY);
        }
        return true;
    }
}
