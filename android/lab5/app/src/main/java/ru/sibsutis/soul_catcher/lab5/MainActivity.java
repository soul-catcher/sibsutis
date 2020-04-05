package ru.sibsutis.soul_catcher.lab5;

import androidx.appcompat.app.AppCompatActivity;

import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteException;
import android.database.sqlite.SQLiteOpenHelper;
import android.os.Bundle;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    private SQLiteDatabase db;
    private Cursor cursor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TableLayout tableLayout = findViewById(R.id.table);
        SQLiteOpenHelper studentsDatabaseHelper = new StudentsDatabaseHelper(this);
        try {
            db = studentsDatabaseHelper.getReadableDatabase();
            cursor = db.query("STUDENTS", new String[]{"NAME", "WEIGHT", "HEIGHT", "AGE"},
                    null, null, null, null, "AGE");
            while (cursor.moveToNext()) {
                TableRow row = new TableRow(this);

                TextView tvName = new TextView(this);
                tvName.setText(cursor.getString(0));

                TextView tvWeight = new TextView(this);
                tvWeight.setText(String.valueOf(cursor.getInt(1)));

                TextView tvHeight = new TextView(this);
                tvHeight.setText(String.valueOf(cursor.getInt(2)));

                TextView tvAge = new TextView(this);
                tvAge.setText(String.valueOf(cursor.getString(3)));

                row.addView(tvName);
                row.addView(tvWeight);
                row.addView(tvHeight);
                row.addView(tvAge);

                tableLayout.addView(row);
            }
        } catch (SQLiteException e) {
            Toast.makeText(this, "Database unavailable", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cursor.close();
        db.close();
    }
}
