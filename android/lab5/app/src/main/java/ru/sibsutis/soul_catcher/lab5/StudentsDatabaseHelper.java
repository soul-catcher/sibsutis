package ru.sibsutis.soul_catcher.lab5;

import android.content.ContentValues;
import android.content.Context;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteException;
import android.database.sqlite.SQLiteOpenHelper;

class StudentsDatabaseHelper extends SQLiteOpenHelper {
    private static final String DB_NAME = "students";
    private static final int DB_VERSION = 1;
    private Context context;

    StudentsDatabaseHelper(Context context) {
        super(context, DB_NAME, null, DB_VERSION);
        this.context = context;
    }

    private static void insertStudent(SQLiteDatabase db, String name, int weight, int height, int age) {
        ContentValues studentValues = new ContentValues();
        studentValues.put("NAME", name);
        studentValues.put("WEIGHT", weight);
        studentValues.put("HEIGHT", height);
        studentValues.put("AGE", age);
        db.insert("STUDENTS", null, studentValues);
    }


    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE STUDENTS (_id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "NAME TEXT, " +
                "WEIGHT INTEGER, " +
                "HEIGHT INTEGER, " +
                "AGE INTEGER);");
        StudentsGenerator sg = new StudentsGenerator(context);
        for (int i = 0; i < 1000; i++) {
            insertStudent(db, sg.genName(), sg.genWeight(), sg.genHeight(), sg.genAge());
        }
    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        throw new SQLiteException("Database cannot be upgraded");
    }
}
