import threading
import tkinter as tk
from tkinter import messagebox, ttk

import sounddevice as sd

from audio_engine import BufferedOutput, HeadphoneMixer, UtteranceCapture
from local_engine import LocalModels, TextDirectionWorker
from voicemeeter_control import ensure_banana_routing


class ScrollColumn(ttk.LabelFrame):
    def __init__(self, parent, title):
        super().__init__(parent, text=title, padding=(8, 6))
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(
            yscrollcommand=self.scroll.set,
            yscrollincrement=24,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._inner_changed)
        self.canvas.bind("<Configure>", self._canvas_changed)
        self.bind_wheel_tree(self)

    def _on_mousewheel(self, event):
        if getattr(event, "delta", 0):
            direction = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(direction * 2, "units")
            return "break"

    def _on_linux_wheel(self, event):
        direction = -1 if event.num == 4 else 1
        self.canvas.yview_scroll(direction * 2, "units")
        return "break"

    def bind_wheel_tree(self, widget):
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_linux_wheel, add="+")
        widget.bind("<Button-5>", self._on_linux_wheel, add="+")
        for child in widget.winfo_children():
            self.bind_wheel_tree(child)

    def _inner_changed(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)

    def _canvas_changed(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def clear(self):
        for child in self.inner.winfo_children():
            child.destroy()


class ManualInterpreterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Live Interpreter MANUAL | UA ↔ EN")
        self.geometry("1280x860")
        self.minsize(1080, 720)

        self.models = None
        self.running = False
        self.captures = []
        self.workers = []
        self.players = []
        self.hp = None
        self.vm = None
        self.original_gain = 0.08

        self.model_status = tk.StringVar(value="Preparing local AI…")
        self.ua_en_status = tk.StringVar(value="waiting for models")
        self.en_ua_status = tk.StringVar(value="waiting for models")
        self.gain_var = tk.DoubleVar(value=0.08)
        self.system_var = tk.StringVar(value="")

        self.devices = list(sd.query_devices())
        self.inputs = self._devices(True)
        self.outputs = self._devices(False)
        self._build_device_maps()

        self._build()
        self._defaults()
        self.after(300, self._load_models_async)

    def _devices(self, input_side):
        result = {}
        hosts = sd.query_hostapis()
        key = "max_input_channels" if input_side else "max_output_channels"
        for idx, dev in enumerate(self.devices):
            if dev[key] > 0:
                host = hosts[dev["hostapi"]]["name"]
                result[f"{idx}: [{host}] {dev['name']}"] = idx
        return result

    @staticmethod
    def _only_devices(mapping, needles):
        result = {}
        for needle in needles:
            preferred = [
                (label, idx) for label, idx in mapping.items()
                if "[Windows WASAPI]" in label and needle.lower() in label.lower()
            ]
            matches = preferred or [
                (label, idx) for label, idx in mapping.items()
                if needle.lower() in label.lower()
            ]
            if matches:
                label, idx = matches[0]
                result[label] = idx
        return result

    def _build_device_maps(self):
        # List devices from the user's machine instead of requiring the
        # original developer's Realtek microphone and JBL headphones.
        self.mic_inputs = dict(self.inputs)
        self.meet_inputs = dict(self.inputs)
        self.headphone_outputs = dict(self.outputs)
        self.meet_outputs = dict(self.outputs)

    @staticmethod
    def _pick(mapping, needles):
        for needle in needles:
            for label in mapping:
                if needle.lower() in label.lower():
                    return label
        return next(iter(mapping), "")

    def _defaults(self):
        self.mic_var.set(self._pick(
            self.mic_inputs,
            ["Microphone Array", "Microphone", "Mic"],
        ))
        self.meet_in_var.set(self._pick(
            self.meet_inputs,
            ["Voicemeeter Out B1", "Voicemeeter VAIO"],
        ))
        self.headphones_var.set(self._pick(
            self.headphone_outputs,
            ["Headphones", "Headset", "Speakers"],
        ))
        self.meet_out_var.set(self._pick(
            self.meet_outputs,
            ["Voicemeeter AUX Input"],
        ))

    def _build(self):
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="Live Interpreter MANUAL",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(
            root,
            text="Текст перекладається автоматично. Голос запускається тільки кнопкою.",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(2, 10))

        ttk.Label(root, text="Local AI:").grid(row=2, column=0, sticky="w")
        ttk.Label(root, textvariable=self.model_status).grid(
            row=2, column=1, columnspan=3, sticky="w"
        )

        self.mic_var = tk.StringVar()
        self.meet_in_var = tk.StringVar()
        self.headphones_var = tk.StringVar()
        self.meet_out_var = tk.StringVar()

        rows = [
            ("Я говорю: фізичний мікрофон", self.mic_var, self.mic_inputs, "mic_combo"),
            ("Вони говорять: Voicemeeter Out B1", self.meet_in_var, self.meet_inputs, "meet_in_combo"),
            ("Я чую: навушники", self.headphones_var, self.headphone_outputs, "headphones_combo"),
            ("Meet чує: Voicemeeter AUX Input", self.meet_out_var, self.meet_outputs, "meet_out_combo"),
        ]
        for row, (label, var, source, attr) in enumerate(rows, 3):
            ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", pady=4)
            combo = ttk.Combobox(
                root, textvariable=var, values=list(source.keys()), state="readonly"
            )
            combo.grid(row=row, column=1, columnspan=3, sticky="ew", padx=(10, 0), pady=4)
            setattr(self, attr, combo)

        ttk.Label(root, text="Оригінальна англійська в навушниках").grid(
            row=7, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Scale(
            root, from_=0.0, to=0.30,
            variable=self.gain_var,
            command=self._gain_changed,
        ).grid(row=7, column=1, sticky="ew", padx=(10, 8), pady=(6, 0))
        ttk.Label(root, text="0–30%").grid(row=7, column=2, sticky="w", pady=(6, 0))

        controls = ttk.Frame(root)
        controls.grid(row=8, column=0, columnspan=4, sticky="ew", pady=10)

        self.start_btn = ttk.Button(
            controls,
            text="▶ START LISTENING",
            command=self.start_interpreter,
            state="disabled",
        )
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(
            controls,
            text="■ STOP",
            command=self.stop_interpreter,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=8)
        ttk.Button(
            controls, text="↻ Refresh audio", command=self.refresh
        ).pack(side="left")
        ttk.Button(
            controls, text="🧹 Очистити картки", command=self._clear_cards
        ).pack(side="left", padx=(8, 0))

        status = ttk.Frame(root)
        status.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        ttk.Label(status, text="Мій канал:").pack(side="left")
        ttk.Label(status, textvariable=self.ua_en_status).pack(side="left", padx=(4, 18))
        ttk.Label(status, text="Їхній канал:").pack(side="left")
        ttk.Label(status, textvariable=self.en_ua_status).pack(side="left", padx=(4, 18))
        ttk.Label(status, textvariable=self.system_var).pack(side="right")

        panels = ttk.Frame(root)
        panels.grid(row=10, column=0, columnspan=4, sticky="nsew")
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(0, weight=1)

        self.incoming = ScrollColumn(
            panels, "ВОНИ → ТИ   English → Українська"
        )
        self.incoming.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.outgoing = ScrollColumn(
            panels, "ТИ → ВОНИ   Українська → English"
        )
        self.outgoing.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ttk.Label(
            root,
            text="Meet: Speaker = Voicemeeter Input   |   Microphone = Voicemeeter Out B2",
            font=("Segoe UI", 10, "bold"),
        ).grid(row=11, column=0, columnspan=4, sticky="w", pady=(8, 0))

        root.columnconfigure(1, weight=1)
        root.columnconfigure(3, weight=1)
        root.rowconfigure(10, weight=1)

    def _gain_changed(self, value):
        self.original_gain = float(value)

    def _set_var(self, var, value):
        self.after(0, lambda: var.set(value))

    def _set_system(self, value):
        self._set_var(self.system_var, value)

    def _load_models_async(self):
        def load():
            try:
                import os
                self.models = LocalModels(os.path.dirname(__file__), self._model_status)
                self.after(0, self._models_ready)
            except Exception as exc:
                self._model_status("FAILED")
                self._set_system(f"Model error: {exc}")
        threading.Thread(target=load, daemon=True).start()

    def _model_status(self, value):
        self._set_var(self.model_status, value)

    def _models_ready(self):
        self.ua_en_status.set("ready")
        self.en_ua_status.set("ready")
        self.start_btn.config(state="normal")
        self._set_system("Local AI ready")

    def refresh(self):
        self.devices = list(sd.query_devices())
        self.inputs = self._devices(True)
        self.outputs = self._devices(False)
        self._build_device_maps()
        for combo, source in (
            (self.mic_combo, self.mic_inputs),
            (self.meet_in_combo, self.meet_inputs),
            (self.headphones_combo, self.headphone_outputs),
            (self.meet_out_combo, self.meet_outputs),
        ):
            combo["values"] = list(source.keys())
        self._defaults()
        self._set_system("Audio devices refreshed")

    def _clear_cards(self):
        self.incoming.clear()
        self.outgoing.clear()

    @staticmethod
    def _make_text(parent, text, height=3):
        box = tk.Text(
            parent,
            height=height,
            wrap="word",
            font=("Segoe UI", 10),
            relief="solid",
            borderwidth=1,
        )
        box.insert("1.0", text)
        return box

    def _add_card(
        self, column, source_label, source, target_label, target,
        action_text, action, secondary_text=None, secondary_action=None
    ):
        card = ttk.Frame(column.inner, padding=8, relief="groove")
        card.pack(fill="x", pady=(0, 8))
        card.columnconfigure(0, weight=1)

        ttk.Label(card, text=source_label, font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            card,
            text=source,
            wraplength=520,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(card, text=target_label, font=("Segoe UI", 9, "bold")).grid(
            row=2, column=0, sticky="w"
        )
        target_box = self._make_text(card, target, height=3)
        target_box.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        footer = ttk.Frame(card)
        footer.grid(row=4, column=0, sticky="ew")
        status_var = tk.StringVar(value="")
        ttk.Label(footer, textvariable=status_var).pack(side="left")

        button = ttk.Button(footer, text=action_text)
        button.pack(side="right")
        button.configure(
            command=lambda: action(target_box, button, status_var)
        )

        if secondary_text and secondary_action:
            secondary_button = ttk.Button(footer, text=secondary_text)
            secondary_button.pack(side="right", padx=(0, 8))
            secondary_button.configure(
                command=lambda: secondary_action(
                    target_box, secondary_button, status_var
                )
            )

        # New cards are created after ScrollColumn initialization, so bind
        # the wheel to every widget inside the card as well.
        column.bind_wheel_tree(card)
        column.after(50, lambda: column.canvas.yview_moveto(1.0))
        return target_box

    def _incoming_result(self, source, translated, elapsed, queued):
        self.after(
            0,
            lambda: self._add_card(
                self.incoming,
                "English",
                source,
                "Український переклад",
                translated,
                "🔊 Озвучити мені",
                self._speak_uk,
            ),
        )

        self._set_system(f"Incoming translated in {elapsed:.2f}s")

    def _outgoing_result(self, source, translated, elapsed, queued):
        self.after(
            0,
            lambda: self._add_card(
                self.outgoing,
                "Українська",
                source,
                "English translation",
                translated,
                "▶ Відправити в Meet",
                self._send_en,
                "🔊 Послухати",
                self._preview_en,
            ),
        )
        self._set_system(f"Your phrase translated in {elapsed:.2f}s")

    def _voice_action(self, text_box, button, status_var, language, output_cb, busy_text):
        text = text_box.get("1.0", "end").strip()
        if not text or self.models is None or output_cb is None:
            return

        button.config(state="disabled")
        status_var.set(busy_text)

        def work():
            try:
                audio = self.models.synthesize(text, language)
                output_cb(audio)
                self.after(0, lambda: status_var.set("✓"))
            except Exception as exc:
                self.after(0, lambda: status_var.set(f"Помилка: {exc}"))
            finally:
                self.after(0, lambda: button.config(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _speak_uk(self, text_box, button, status_var):
        self._voice_action(
            text_box, button, status_var,
            "uk",
            self.hp.push_translation_48 if self.hp else None,
            "Готую голос…",
        )

    def _preview_en(self, text_box, button, status_var):
        self._voice_action(
            text_box, button, status_var,
            "en",
            self.hp.push_translation_48 if self.hp else None,
            "Готую прослуховування…",
        )

    def _send_en(self, text_box, button, status_var):
        self._voice_action(
            text_box, button, status_var,
            "en",
            self.vm.push_pcm48 if self.vm else None,
            "Готую та відправляю…",
        )

    def _error(self, message):
        self._set_system(message)

    def start_interpreter(self):
        if self.running or self.models is None:
            return

        routing_ok, routing_message = ensure_banana_routing()
        self._set_system(routing_message)
        if not routing_ok:
            messagebox.showerror("VoiceMeeter", routing_message)
            return

        try:
            mic = self.inputs[self.mic_var.get()]
            meet_in = self.inputs[self.meet_in_var.get()]
            headphones = self.outputs[self.headphones_var.get()]
            meet_out = self.outputs[self.meet_out_var.get()]
        except KeyError:
            messagebox.showerror("Audio", "Оберіть усі 4 аудіопристрої.")
            return

        try:
            self.hp = HeadphoneMixer(headphones, lambda: self.original_gain)
            self.vm = BufferedOutput(meet_out)
            self.hp.start()
            self.vm.start()

            ua_worker = TextDirectionWorker(
                "UA→EN",
                self.models,
                "uk",
                "en",
                lambda s: self._set_var(self.ua_en_status, s),
                self._outgoing_result,
                self._error,
            )
            en_worker = TextDirectionWorker(
                "EN→UA",
                self.models,
                "en",
                "uk",
                lambda s: self._set_var(self.en_ua_status, s),
                self._incoming_result,
                self._error,
            )

            mic_cap = UtteranceCapture(
                mic,
                ua_worker.submit,
                min_threshold=120.0,
                pre_roll_ms=340,
                start_voice_ms=60,
                end_silence_ms=820,
                vad_aggressiveness=1,
                min_utterance_ms=300,
            )
            meet_cap = UtteranceCapture(
                meet_in,
                en_worker.submit,
                passthrough=self.hp.push_original_48,
                min_threshold=70.0,
                pre_roll_ms=340,
                start_voice_ms=60,
                end_silence_ms=820,
                vad_aggressiveness=1,
                min_utterance_ms=300,
            )

            mic_cap.start()
            meet_cap.start()

            self.players = [self.hp, self.vm]
            self.workers = [ua_worker, en_worker]
            self.captures = [mic_cap, meet_cap]
            self.running = True

            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.ua_en_status.set("listening")
            self.en_ua_status.set("listening")
            self._set_system("Manual mode: voice only by button")
        except Exception as exc:
            self._cleanup()
            messagebox.showerror("Start failed", str(exc))

    def _cleanup(self):
        for item in self.captures + self.workers + self.players:
            try:
                item.stop()
            except Exception:
                pass
        self.captures = []
        self.workers = []
        self.players = []
        self.hp = None
        self.vm = None

    def stop_interpreter(self):
        if not self.running:
            return
        self._cleanup()
        self.running = False
        self.ua_en_status.set("ready")
        self.en_ua_status.set("ready")
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._set_system("Stopped")

    def destroy(self):
        self._cleanup()
        super().destroy()


if __name__ == "__main__":
    ManualInterpreterApp().mainloop()
