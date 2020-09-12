package ru.sibsutis.soul_catcher.lab3;

import android.os.Bundle;
import android.os.Handler;

import androidx.appcompat.app.AppCompatActivity;

import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {

    private BugView view;
    private Handler handler;
    private final static int interval = 1000 / 60;  // 60 fps


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        view = new BugView(this);
        setContentView(view);

        handler = new Handler();
        Timer timer = new Timer();
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        view.invalidate();
                    }
                });
            }
        }, 0, interval);
    }
}