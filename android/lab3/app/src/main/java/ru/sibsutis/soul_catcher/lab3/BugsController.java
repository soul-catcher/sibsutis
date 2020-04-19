package ru.sibsutis.soul_catcher.lab3;

import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.media.MediaPlayer;
import android.view.View;

import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

class BugsController {
    private View view;
    private ArrayList<Bug> bugsList;
    private final int bugsCount;
    int points;

    private void removeDeads() {
        ArrayList<Bug> newBugsList = new ArrayList<>(5);
        for (Bug bug : bugsList) {
            if (bug.alive) {
                newBugsList.add(bug);
            }
        }
        bugsList = newBugsList;
    }

    void drawBugs(Canvas canvas) {
        for (Bug bug : bugsList) {
            canvas.drawBitmap(bug.texture, bug.matrix, null);
        }
    }

    private void createBug() {
        Bug bug = new Bug();
        switch (ThreadLocalRandom.current().nextInt(0, 3)) {
            case 0:
                bug.texture = BitmapFactory.decodeResource(view.getResources(), R.drawable.bug1);
                break;
            case 1:
                bug.texture = BitmapFactory.decodeResource(view.getResources(), R.drawable.bug2);
                break;
            case 2:
                bug.texture = BitmapFactory.decodeResource(view.getResources(), R.drawable.bug3);
                break;
        }
        bugsList.add(bug);
        bug.matrix.setRotate(0, bug.texture.getWidth() / 2, bug.texture.getHeight() / 2);
        bug.matrix.reset();
        bug.p = 0;
        bug.isRunning = false;
        float ty, tx;
        int temp = (int) Math.floor(Math.random() * 4);
        switch (temp) {
            case 0:
                ty = (float) Math.random() * view.getHeight();
                bug.x = 0f;
                bug.y = ty;
                break;
            case 1:
                ty = (float) Math.random() * view.getHeight();
                bug.x = (float) view.getWidth();
                bug.y = ty;
                break;
            case 2:
                tx = (float) Math.random() * view.getWidth();
                bug.x = tx;
                bug.y = 0f;
                break;
            case 3:
                tx = (float) Math.random() * view.getWidth();
                bug.x = tx;
                bug.y = (float) view.getHeight();
                break;
        }
        bug.matrix.postTranslate(bug.x, bug.y);
    }

    private void handleBug(Bug bug) {
        if (!bug.isRunning) {
            bug.destX = (float) Math.random() * view.getWidth();
            bug.destY = (float) Math.random() * view.getHeight();
            bug.stepX = (bug.destX - bug.x) / 150;
            bug.stepY = (bug.destY - bug.y) / 150;
            Integer tp;
            if (bug.x <= bug.destX && bug.y >= bug.destY)
                tp = (int) Math.floor(Math.toDegrees(Math.atan(Math.abs(bug.x - bug.destX) / Math.abs(bug.y - bug.destY))));
            else if (bug.x <= bug.destX && bug.y <= bug.destY)
                tp = 90 + (int) Math.floor(Math.toDegrees(Math.atan(Math.abs(bug.y - bug.destY) / Math.abs(bug.x - bug.destX))));
            else if (bug.x >= bug.destX && bug.y <= bug.destY)
                tp = 180 + (int) Math.floor(Math.toDegrees(Math.atan(Math.abs(bug.x - bug.destX) / Math.abs(bug.y - bug.destY))));
            else
                tp = 270 + (int) Math.floor(Math.toDegrees(Math.atan(Math.abs(bug.y - bug.destY) / Math.abs(bug.x - bug.destX))));
            bug.matrix.preRotate(tp - bug.p, bug.texture.getWidth() / 2, bug.texture.getHeight() / 2);
            bug.p = tp;
            bug.isRunning = true;
        } else {
            if (Math.abs(bug.x - bug.destX) < 0.1 &&
                    Math.abs(bug.y - bug.destY) < 0.1)
                bug.isRunning = false;

            bug.matrix.postTranslate(bug.stepX, bug.stepY);
            bug.x += bug.stepX;
            bug.y += bug.stepY;
        }
    }

    void touchEvent(float x, float y) {
        boolean hit = false;
        for (Bug bug : bugsList) {
            if (Math.abs(bug.x - x + 60) < 140 && Math.abs(bug.y - y + 60) < 150) {
                MediaPlayer.create(view.getContext(), R.raw.hit).start();
                bug.die();
                points += 50;
                hit = true;
                break;
            }
        }
        if (!hit) {
            MediaPlayer.create(view.getContext(), R.raw.miss).start();
            points -= 10;
        }
    }


    BugsController(int bugsCount, View view) {
        points = 0;
        this.view = view;
        this.bugsCount = bugsCount;
        bugsList = new ArrayList<>(5);
    }

    void update() {
        removeDeads();
        while (bugsList.size() < bugsCount) {
            createBug();
        }
        for (final Bug bug : bugsList) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    handleBug(bug);
                }
            }).start();
        }
    }

}
