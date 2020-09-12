package ru.sibsutis.soul_catcher.lab5;

import android.content.Context;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

class StudentsGenerator {
    private List<String> names = new ArrayList<>();
    private Random random = new Random();

    StudentsGenerator(Context context) {
        try {
            Scanner fileScanner = new Scanner(context.getAssets().open("names.txt"));
            while (fileScanner.hasNextLine()) {
                names.add(fileScanner.nextLine());
            }
            fileScanner.close();
        } catch (IOException e) {
            Toast.makeText(context, "Cannot open file for generate names", Toast.LENGTH_SHORT).show();
        }

    }

    String genName() {
        return names.get(random.nextInt(names.size()));
    }

    int genWeight() {
        return random.nextInt(40) + 50;
    }

    int genHeight() {
        return random.nextInt(50) + 150;
    }

    int genAge() {
        return random.nextInt(10) + 17;
    }
}
