from datetime import datetime, timedelta
from pathlib import Path
import csv
import html


BASE_DIR = Path(__file__).resolve().parents[2]

LOG_DIR = BASE_DIR / "data" / "session_logs"
REPORT_DIR = BASE_DIR / "reports"

LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class SessionLogger:

    def __init__(self, started_at=None):

        # predict.py may pass either:
        # datetime object OR Unix timestamp.

        if isinstance(started_at, datetime):
            self.session_start = started_at

        elif isinstance(started_at, (int, float)):
            self.session_start = datetime.fromtimestamp(
                started_at
            )

        else:
            self.session_start = datetime.now()

        self.observations = []


    def _convert_timestamp(self, value):

        if isinstance(value, datetime):
            return value

        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)

        return datetime.now()


    def record(self, *args, state=None, **kwargs):

        """
        Compatible with the existing predict.py.

        Handles calls containing:
        - Unix timestamps
        - datetime objects
        - elapsed seconds
        - state keyword
        """

        timestamp = kwargs.get("timestamp")

        elapsed_seconds = kwargs.get(
            "elapsed_seconds"
        )


        # Find timestamp / elapsed values from positional args.

        numeric_values = []

        for arg in args:

            if isinstance(arg, datetime):

                timestamp = arg

            elif isinstance(arg, (int, float)):

                numeric_values.append(arg)


        # If timestamp was supplied as a keyword,
        # convert it properly.

        if timestamp is not None:

            timestamp = self._convert_timestamp(
                timestamp
            )


        # If no timestamp was supplied,
        # use current time.

        if timestamp is None:

            timestamp = datetime.now()


        # The existing predict.py may pass the elapsed
        # value positionally.

        if elapsed_seconds is None:

            if numeric_values:

                # If timestamp is already provided separately,
                # the remaining numeric value is elapsed time.

                if (
                    len(numeric_values) >= 2
                    and isinstance(
                        kwargs.get("timestamp"),
                        (int, float)
                    )
                ):

                    elapsed_seconds = (
                        numeric_values[-1]
                    )

                else:

                    elapsed_seconds = (
                        numeric_values[-1]
                    )

            else:

                elapsed_seconds = (
                    timestamp
                    - self.session_start
                ).total_seconds()


        # State can be supplied through state= or status=.

        if state is None:

            state = kwargs.get(
                "status",
                "focused"
            )


        self.observations.append(
            {
                "timestamp": timestamp,
                "elapsed_seconds": float(
                    elapsed_seconds
                ),
                "state": str(state),
            }
        )


    def log(self, state, elapsed_seconds):

        self.record(
            elapsed_seconds,
            state=state
        )


    def save(self, *args, **kwargs):

        """
        Saves the CSV and HTML dashboard.

        Extra arguments are accepted because
        predict.py may pass additional values.
        """

        if not self.observations:

            return None, None


        session_id = (
            self.session_start.strftime(
                "%Y%m%d_%H%M%S"
            )
        )


        csv_path = (
            LOG_DIR
            / f"session_{session_id}.csv"
        )


        report_path = (
            REPORT_DIR
            / f"session_{session_id}_summary.html"
        )


        self._save_csv(csv_path)


        stats = self.calculate_stats()

        history = self.build_daily_history()

        streaks = self.calculate_streaks(
            history
        )

        calendar = self.weekly_calendar(
            history
        )

        motivation = (
            self.motivational_message(
                stats,
                streaks
            )
        )


        dashboard = self.build_dashboard(
            stats=stats,
            history=history,
            streaks=streaks,
            calendar=calendar,
            motivation=motivation,
            csv_filename=csv_path.name,
        )


        report_path.write_text(
            dashboard,
            encoding="utf-8"
        )


        return csv_path, report_path


    def _save_csv(self, path):

        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)


            writer.writerow(
                [
                    "timestamp",
                    "elapsed_seconds",
                    "state",
                ]
            )


            for item in self.observations:

                timestamp = item["timestamp"]


                if isinstance(
                    timestamp,
                    datetime
                ):

                    timestamp = (
                        timestamp.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    )


                writer.writerow(
                    [
                        timestamp,
                        round(
                            item[
                                "elapsed_seconds"
                            ],
                            2
                        ),
                        item["state"],
                    ]
                )


    def calculate_stats(self):

        state_times = {
            "focused": 0.0,
            "phone_use": 0.0,
            "drowsy": 0.0,
            "distracted": 0.0,
            "on_break": 0.0,
        }


        if not self.observations:

            return {
                "total_seconds": 0,
                "focused_seconds": 0,
                "phone_seconds": 0,
                "drowsy_seconds": 0,
                "distracted_seconds": 0,
                "break_seconds": 0,
                "concentration": 0,
                "break_count": 0,
                "start_time": self.session_start,
                "end_time": self.session_start,
            }


        observations = sorted(
            self.observations,
            key=lambda item: item["timestamp"]
        )


        break_count = 0


        for i, item in enumerate(
            observations
        ):

            state = item["state"]


            if state == "on_break":

                if (
                    i == 0
                    or observations[
                        i - 1
                    ]["state"]
                    != "on_break"
                ):

                    break_count += 1


            if i < len(observations) - 1:

                current_time = (
                    observations[i][
                        "timestamp"
                    ]
                )

                next_time = (
                    observations[i + 1][
                        "timestamp"
                    ]
                )


                duration = (
                    next_time
                    - current_time
                ).total_seconds()


                if duration < 0:
                    duration = 0


                # Prevent accidental huge gaps.

                if duration > 2:
                    duration = 2


                if state in state_times:

                    state_times[state] += (
                        duration
                    )


        start_time = observations[0][
            "timestamp"
        ]

        end_time = observations[-1][
            "timestamp"
        ]


        total_seconds = max(
            0,
            (
                end_time
                - start_time
            ).total_seconds()
        )


        focused_seconds = (
            state_times["focused"]
        )

        phone_seconds = (
            state_times["phone_use"]
        )

        drowsy_seconds = (
            state_times["drowsy"]
        )

        distracted_seconds = (
            state_times["distracted"]
        )

        break_seconds = (
            state_times["on_break"]
        )


        active_seconds = max(
            0,
            total_seconds
            - break_seconds
        )


        if active_seconds > 0:

            concentration = (
                focused_seconds
                / active_seconds
            ) * 100

        else:

            concentration = 0


        concentration = max(
            0,
            min(
                100,
                concentration
            )
        )


        return {
            "total_seconds": total_seconds,
            "focused_seconds": focused_seconds,
            "phone_seconds": phone_seconds,
            "drowsy_seconds": drowsy_seconds,
            "distracted_seconds": distracted_seconds,
            "break_seconds": break_seconds,
            "concentration": concentration,
            "break_count": break_count,
            "start_time": start_time,
            "end_time": end_time,
        }


    def read_previous_sessions(self):

        sessions = []


        for csv_file in sorted(
            LOG_DIR.glob(
                "session_*.csv"
            )
        ):

            try:

                rows = []


                with csv_file.open(
                    "r",
                    encoding="utf-8",
                    newline=""
                ) as file:

                    reader = csv.DictReader(
                        file
                    )

                    for row in reader:

                        rows.append(row)


                if not rows:
                    continue


                date_value = None


                for row in rows:

                    timestamp = row.get(
                        "timestamp",
                        ""
                    )


                    try:

                        date_value = (
                            datetime.strptime(
                                timestamp,
                                "%Y-%m-%d %H:%M:%S"
                            ).date()
                        )

                        break

                    except ValueError:

                        continue


                if date_value is None:
                    continue


                focused = 0

                active = 0


                for row in rows:

                    state = row.get(
                        "state",
                        ""
                    )


                    if state == "focused":

                        focused += 1


                    if state != "on_break":

                        active += 1


                if active > 0:

                    concentration = (
                        focused
                        / active
                    ) * 100

                else:

                    concentration = 0


                sessions.append(
                    {
                        "date": date_value,
                        "concentration": round(
                            concentration,
                            1
                        ),
                        "file": csv_file.name,
                    }
                )


            except Exception:

                continue


        return sessions


    def build_daily_history(self):

        history = (
            self.read_previous_sessions()
        )


        current_date = (
            self.session_start.date()
        )


        current_stats = (
            self.calculate_stats()
        )


        found_current = False


        for item in history:

            if (
                item["date"]
                == current_date
            ):

                found_current = True

                break


        if not found_current:

            history.append(
                {
                    "date": current_date,
                    "concentration": round(
                        current_stats[
                            "concentration"
                        ],
                        1
                    ),
                    "file": "",
                }
            )


        unique = {}


        for item in history:

            unique[
                item["date"]
            ] = item


        return sorted(
            unique.values(),
            key=lambda item: item["date"]
        )


    def get_study_dates(
        self,
        history
    ):

        return sorted(
            {
                item["date"]
                for item in history
                if item[
                    "concentration"
                ] > 0
            }
        )


    def calculate_streaks(
        self,
        history
    ):

        dates = (
            self.get_study_dates(
                history
            )
        )


        if not dates:

            return {
                "current": 0,
                "best": 0,
                "study_days": 0,
            }


        date_set = set(dates)


        best = 1

        running = 1


        for i in range(
            1,
            len(dates)
        ):

            if (
                dates[i]
                == dates[i - 1]
                + timedelta(days=1)
            ):

                running += 1

            else:

                running = 1


            best = max(
                best,
                running
            )


        today = datetime.now().date()


        if today in date_set:

            current_date = today

        elif (
            today
            - timedelta(days=1)
            in date_set
        ):

            current_date = (
                today
                - timedelta(days=1)
            )

        else:

            current_date = None


        current = 0


        if current_date is not None:

            current = 1

            check_date = (
                current_date
                - timedelta(days=1)
            )


            while (
                check_date
                in date_set
            ):

                current += 1

                check_date -= timedelta(
                    days=1
                )


        return {
            "current": current,
            "best": best,
            "study_days": len(dates),
        }


    def weekly_calendar(
        self,
        history
    ):

        today = datetime.now().date()


        monday = (
            today
            - timedelta(
                days=today.weekday()
            )
        )


        study_dates = set(
            self.get_study_dates(
                history
            )
        )


        days = []


        for i in range(7):

            day = (
                monday
                + timedelta(days=i)
            )


            days.append(
                {
                    "date": day,
                    "label": day.strftime(
                        "%a"
                    ),
                    "number": day.day,
                    "studied": (
                        day
                        in study_dates
                    ),
                    "today": (
                        day == today
                    ),
                }
            )


        return days


    def build_chart(
        self,
        history
    ):

        if not history:

            return """
            <div class="empty-chart">
                No study history available yet.
            </div>
            """


        width = 760

        height = 280

        padding = 45


        usable_width = (
            width
            - padding * 2
        )

        usable_height = (
            height
            - padding * 2
        )


        count = len(history)


        points = []


        for i, item in enumerate(
            history
        ):

            if count == 1:

                x = width / 2

            else:

                x = (
                    padding
                    + (
                        i
                        / (count - 1)
                    )
                    * usable_width
                )


            value = max(
                0,
                min(
                    100,
                    item[
                        "concentration"
                    ]
                )
            )


            y = (
                padding
                + (
                    (100 - value)
                    / 100
                )
                * usable_height
            )


            points.append(
                (
                    x,
                    y,
                    value,
                    item["date"]
                )
            )


        polyline = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y, _, _
            in points
        )


        circles = ""


        for (
            x,
            y,
            value,
            date_value
        ) in points:

            circles += f"""
            <circle
                cx="{x:.1f}"
                cy="{y:.1f}"
                r="5"
                class="chart-point">

                <title>
                    {html.escape(
                        str(date_value)
                    )}: {value:.1f}%
                </title>

            </circle>
            """


        return f"""
        <div class="chart-wrapper">

            <svg
                viewBox="0 0 {width} {height}"
                class="chart">

                <line
                    x1="{padding}"
                    y1="{padding}"
                    x2="{padding}"
                    y2="{height - padding}"
                    class="axis"/>

                <line
                    x1="{padding}"
                    y1="{height - padding}"
                    x2="{width - padding}"
                    y2="{height - padding}"
                    class="axis"/>

                <line
                    x1="{padding}"
                    y1="{padding}"
                    x2="{width - padding}"
                    y2="{padding}"
                    class="grid-line"/>

                <line
                    x1="{padding}"
                    y1="{height / 2}"
                    x2="{width - padding}"
                    y2="{height / 2}"
                    class="grid-line"/>

                <polyline
                    points="{polyline}"
                    class="chart-line"/>

                {circles}

                <text
                    x="10"
                    y="{padding + 5}"
                    class="axis-label">
                    100%
                </text>

                <text
                    x="20"
                    y="{height / 2 + 5}"
                    class="axis-label">
                    50%
                </text>

                <text
                    x="30"
                    y="{height - padding + 5}"
                    class="axis-label">
                    0%
                </text>

            </svg>

        </div>
        """


    def motivational_message(
        self,
        stats,
        streaks
    ):

        concentration = (
            stats["concentration"]
        )

        current = (
            streaks["current"]
        )

        best = (
            streaks["best"]
        )


        if current == 0 and best > 0:

            return (
                f"Your streak ended at "
                f"{best} days. "
                "Missing one day doesn't "
                "erase your progress. "
                f"Your previous best: "
                f"{best} days. "
                "Ready to start a new streak?"
            )


        if current == 1:

            return (
                "Welcome back! "
                "Let's continue from here. "
                "One focused day is the "
                "beginning of a new streak."
            )


        if concentration >= 90:

            return (
                "Excellent focus! "
                "You stayed highly "
                "concentrated throughout "
                "this session. "
                "Keep building the habit."
            )


        if concentration >= 75:

            return (
                "Great work! "
                "Your concentration was strong. "
                "Keep going and make your "
                "next session even better."
            )


        if concentration >= 50:

            return (
                "Good effort! "
                "There is room to improve "
                "your focus. "
                "Try reducing distractions "
                "during your next session."
            )


        return (
            "Every session is practice. "
            "Start small, reduce "
            "distractions, and build "
            "your focus step by step."
        )


    @staticmethod
    def format_time(seconds):

        seconds = max(
            0,
            int(round(seconds))
        )


        hours = (
            seconds // 3600
        )

        minutes = (
            seconds % 3600
        ) // 60

        secs = (
            seconds % 60
        )


        if hours > 0:

            return (
                f"{hours}h "
                f"{minutes}m"
            )


        if minutes > 0:

            return (
                f"{minutes}m "
                f"{secs}s"
            )


        return f"{secs}s"


    def build_dashboard(
        self,
        stats,
        history,
        streaks,
        calendar,
        motivation,
        csv_filename
    ):

        concentration = (
            stats["concentration"]
        )


        date_text = (
            stats["start_time"]
            .strftime(
                "%d %B %Y"
            )
        )


        start_text = (
            stats["start_time"]
            .strftime(
                "%I:%M %p"
            )
        )


        end_text = (
            stats["end_time"]
            .strftime(
                "%I:%M %p"
            )
        )


        calendar_html = ""


        for day in calendar:

            classes = [
                "calendar-day"
            ]


            if day["studied"]:

                classes.append(
                    "studied"
                )


            if day["today"]:

                classes.append(
                    "today"
                )


            status = (
                "✓"
                if day["studied"]
                else "—"
            )


            calendar_html += f"""
            <div class="{' '.join(classes)}">

                <div class="day-name">
                    {html.escape(
                        day["label"]
                    )}
                </div>

                <div class="day-number">
                    {day["number"]}
                </div>

                <div class="day-status">
                    {status}
                </div>

            </div>
            """


        chart = self.build_chart(
            history
        )


        csv_link = (
            "../data/session_logs/"
            + html.escape(
                csv_filename
            )
        )


        active_time = max(
            0,
            stats["total_seconds"]
            - stats["break_seconds"]
        )


        return f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1.0">

<title>
    StudyVision Dashboard
</title>


<style>

* {{
    box-sizing: border-box;
}}


body {{

    margin: 0;

    min-height: 100vh;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef2ff 0%,
            #fdf2f8 45%,
            #ecfeff 100%
        );

    color: #172033;
}}


.container {{

    width: min(
        1180px,
        94%
    );

    margin: 0 auto;

    padding:
        35px 0 50px;
}}


.header {{

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed,
            #db2777
        );

    border-radius: 28px;

    padding: 35px;

    color: white;

    box-shadow:
        0 20px 50px
        rgba(
            79,
            70,
            229,
            0.22
        );

    margin-bottom: 25px;
}}


.header h1 {{

    margin:
        0 0 8px;

    font-size: 34px;
}}


.header p {{

    margin: 0;

    opacity: 0.9;

    font-size: 16px;
}}


.header-info {{

    display: flex;

    flex-wrap: wrap;

    gap: 18px;

    margin-top: 25px;
}}


.header-pill {{

    background:
        rgba(
            255,
            255,
            255,
            0.18
        );

    padding:
        10px 16px;

    border-radius: 999px;
}}


.grid {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                220px,
                1fr
            )
        );

    gap: 18px;

    margin-bottom: 25px;
}}


.card {{

    background:
        rgba(
            255,
            255,
            255,
            0.88
        );

    border-radius: 22px;

    padding: 23px;

    box-shadow:
        0 10px 30px
        rgba(
            30,
            41,
            59,
            0.08
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            0.8
        );
}}


.card h3 {{

    margin:
        0 0 10px;

    font-size: 15px;

    color: #64748b;
}}


.metric {{

    font-size: 30px;

    font-weight: 800;
}}


.focus-card {{

    background:
        linear-gradient(
            135deg,
            #dcfce7,
            #d1fae5
        );
}}


.phone-card {{

    background:
        linear-gradient(
            135deg,
            #fef3c7,
            #fde68a
        );
}}


.drowsy-card {{

    background:
        linear-gradient(
            135deg,
            #ede9fe,
            #ddd6fe
        );
}}


.distracted-card {{

    background:
        linear-gradient(
            135deg,
            #fee2e2,
            #fecaca
        );
}}


.break-card {{

    background:
        linear-gradient(
            135deg,
            #dbeafe,
            #bfdbfe
        );
}}


.section {{

    background:
        rgba(
            255,
            255,
            255,
            0.90
        );

    border-radius: 25px;

    padding: 28px;

    margin-bottom: 25px;

    box-shadow:
        0 10px 30px
        rgba(
            30,
            41,
            59,
            0.07
        );
}}


.section h2 {{

    margin-top: 0;

    margin-bottom: 20px;
}}


.streak-grid {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                200px,
                1fr
            )
        );

    gap: 18px;
}}


.streak {{

    padding: 25px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #fff7ed,
            #ffedd5
        );
}}


.streak .number {{

    font-size: 38px;

    font-weight: 900;
}}


.streak .label {{

    color: #64748b;

    margin-top: 5px;
}}


.calendar {{

    display: grid;

    grid-template-columns:
        repeat(
            7,
            1fr
        );

    gap: 10px;
}}


.calendar-day {{

    text-align: center;

    padding:
        15px 8px;

    border-radius: 15px;

    background: #f1f5f9;
}}


.calendar-day.studied {{

    background:
        linear-gradient(
            135deg,
            #dcfce7,
            #bbf7d0
        );
}}


.calendar-day.today {{

    outline:
        3px solid
        #8b5cf6;
}}


.day-name {{

    font-size: 12px;

    color: #64748b;

    font-weight: bold;
}}


.day-number {{

    font-size: 24px;

    font-weight: 800;

    margin:
        6px 0;
}}


.day-status {{

    font-weight: bold;
}}


.chart-wrapper {{

    width: 100%;

    overflow-x: auto;
}}


.chart {{

    min-width: 760px;

    width: 100%;

    height: 300px;
}}


.axis {{

    stroke: #94a3b8;

    stroke-width: 1.5;
}}


.grid-line {{

    stroke: #cbd5e1;

    stroke-width: 1;

    stroke-dasharray: 5 5;
}}


.chart-line {{

    fill: none;

    stroke: #6366f1;

    stroke-width: 4;

    stroke-linecap: round;

    stroke-linejoin: round;
}}


.chart-point {{

    fill: #7c3aed;

    stroke: white;

    stroke-width: 2;
}}


.axis-label {{

    font-size: 12px;

    fill: #64748b;
}}


.motivation {{

    background:
        linear-gradient(
            135deg,
            #fef3c7,
            #fce7f3,
            #ede9fe
        );

    border-radius: 22px;

    padding: 25px;

    font-size: 18px;

    line-height: 1.6;

    font-weight: 600;
}}


.details {{

    display: grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                220px,
                1fr
            )
        );

    gap: 15px;
}}


.detail {{

    background: #f8fafc;

    padding: 17px;

    border-radius: 15px;
}}


.detail-label {{

    color: #64748b;

    font-size: 13px;

    margin-bottom: 5px;
}}


.detail-value {{

    font-weight: 700;
}}


.buttons {{

    display: flex;

    flex-wrap: wrap;

    gap: 12px;

    margin-top: 20px;
}}


.button {{

    display: inline-block;

    text-decoration: none;

    border: none;

    cursor: pointer;

    padding:
        13px 19px;

    border-radius: 12px;

    background: #4f46e5;

    color: white;

    font-weight: 700;
}}


.button.secondary {{

    background: #e2e8f0;

    color: #172033;
}}


.footer {{

    text-align: center;

    color: #64748b;

    padding: 15px;
}}


@media(max-width: 600px) {{

    .header {{
        padding: 25px;
    }}

    .header h1 {{
        font-size: 27px;
    }}

    .calendar {{
        gap: 5px;
    }}

    .calendar-day {{
        padding:
            10px 3px;
    }}

}}

</style>

</head>


<body>

<div class="container">


    <div class="header">

        <h1>
            🎓 StudyVision
        </h1>

        <p>
            Student Study Performance Dashboard
        </p>


        <div class="header-info">

            <div class="header-pill">
                📅 {html.escape(
                    date_text
                )}
            </div>

            <div class="header-pill">
                ▶️ {html.escape(
                    start_text
                )}
            </div>

            <div class="header-pill">
                ⏹️ {html.escape(
                    end_text
                )}
            </div>

        </div>

    </div>


    <div class="grid">


        <div class="card focus-card">

            <h3>
                🎯 Concentration
            </h3>

            <div class="metric">
                {concentration:.1f}%
            </div>

        </div>


        <div class="card">

            <h3>
                ⏱️ Total Session
            </h3>

            <div class="metric">
                {self.format_time(
                    stats[
                        "total_seconds"
                    ]
                )}
            </div>

        </div>


        <div class="card focus-card">

            <h3>
                🟢 Focused Time
            </h3>

            <div class="metric">
                {self.format_time(
                    stats[
                        "focused_seconds"
                    ]
                )}
            </div>

        </div>


        <div class="card phone-card">

            <h3>
                📱 Phone Use
            </h3>

            <div class="metric">
                {self.format_time(
                    stats[
                        "phone_seconds"
                    ]
                )}
            </div>

        </div>


        <div class="card drowsy-card">

            <h3>
                😴 Drowsy Time
            </h3>

            <div class="metric">
                {self.format_time(
                    stats[
                        "drowsy_seconds"
                    ]
                )}
            </div>

        </div>


        <div class="card distracted-card">

            <h3>
                👤 Distracted Time
            </h3>

            <div class="metric">
                {self.format_time(
                    stats[
                        "distracted_seconds"
                    ]
                )}
            </div>

        </div>


        <div class="card break-card">

            <h3>
                ☕ Break Time
            </h3>

            <div class="metric">
                {self.format_time(
                    stats[
                        "break_seconds"
                    ]
                )}
            </div>

        </div>


        <div class="card">

            <h3>
                ☕ Break Count
            </h3>

            <div class="metric">
                {stats[
                    "break_count"
                ]}
            </div>

        </div>


    </div>


    <div class="section">

        <h2>
            🔥 Study Streaks
        </h2>


        <div class="streak-grid">


            <div class="streak">

                <div class="number">
                    🔥
                    {streaks[
                        "current"
                    ]}
                </div>

                <div class="label">
                    Current Streak
                </div>

            </div>


            <div class="streak">

                <div class="number">
                    🏆
                    {streaks[
                        "best"
                    ]}
                </div>

                <div class="label">
                    Best Streak
                </div>

            </div>


            <div class="streak">

                <div class="number">
                    📆
                    {streaks[
                        "study_days"
                    ]}
                </div>

                <div class="label">
                    Study Days
                </div>

            </div>


        </div>

    </div>


    <div class="section">

        <h2>
            📅 This Week
        </h2>


        <div class="calendar">

            {calendar_html}

        </div>

    </div>


    <div class="section">

        <h2>
            📈 Daily Concentration
        </h2>

        {chart}

    </div>


    <div class="section">

        <h2>
            💬 Your Progress
        </h2>


        <div class="motivation">

            {html.escape(
                motivation
            )}

        </div>

    </div>


    <div class="section">

        <h2>
            📝 Session Details
        </h2>


        <div class="details">


            <div class="detail">

                <div class="detail-label">
                    Date
                </div>

                <div class="detail-value">
                    {html.escape(
                        date_text
                    )}
                </div>

            </div>


            <div class="detail">

                <div class="detail-label">
                    Start Time
                </div>

                <div class="detail-value">
                    {html.escape(
                        start_text
                    )}
                </div>

            </div>


            <div class="detail">

                <div class="detail-label">
                    End Time
                </div>

                <div class="detail-value">
                    {html.escape(
                        end_text
                    )}
                </div>

            </div>


            <div class="detail">

                <div class="detail-label">
                    Total Active Study Time
                </div>

                <div class="detail-value">
                    {self.format_time(
                        active_time
                    )}
                </div>

            </div>


        </div>


        <div class="buttons">


            <a
                class="button"
                href="{csv_link}"
                download>

                ⬇️ Download Session CSV

            </a>


            <button
                class="button secondary"
                onclick="window.print()">

                🖨️ Print Dashboard

            </button>


        </div>

    </div>


    <div class="footer">

        StudyVision •
        Keep learning,
        keep improving 🚀

    </div>


</div>

</body>

</html>
"""


if __name__ == "__main__":

    print(
        "StudyVision session report "
        "module loaded successfully."
    )