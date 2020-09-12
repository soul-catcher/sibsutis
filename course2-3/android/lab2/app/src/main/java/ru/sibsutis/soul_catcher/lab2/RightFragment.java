package ru.sibsutis.soul_catcher.lab2;

import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;


public class RightFragment extends Fragment {

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        TableLayout table = new TableLayout(getActivity());

        for (int i = 1; i <= 10; i++) {
            TableRow row = new TableRow(getActivity());
            for (int j = 1; j <= 10; j++) {
                TextView tv = new TextView(getActivity());
                tv.setText(String.valueOf(i * j));
                row.addView(tv);
            }
            table.addView(row);
        }
        return table;
    }
}
