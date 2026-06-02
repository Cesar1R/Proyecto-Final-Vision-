import ffmpeg
import os

def preprocess_video(input_path, output_path, start_time, duration):
    """
    Preprocesa un video para el modelo Action-Slot.
    
    Argumentos:
    input_path (str): Ruta al video original.
    output_path (str): Ruta donde se guardará el video procesado.
    start_time (str/int): Tiempo de inicio (ej. "00:00:10" o segundos).
    duration (int): Duración del clip en segundos.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo {input_path} no existe.")

    try:
        # la entrada y el recorte temporal (ss y t)
        input_stream = ffmpeg.input(input_path, ss=start_time, t=duration)
        
        # aplicar la escala
        video_stream = input_stream.video
        video_stream = ffmpeg.filter(video_stream, 'scale', 1280, 720)
        video_stream = ffmpeg.filter(video_stream, 'fps', fps=30)
        
        # quitamos el audio para la ssalida (-an) y con usamos el codec H.264
        output_stream = ffmpeg.output(
            video_stream, 
            output_path, 
            vcodec='libx264', 
            crf=23,
            pix_fmt='yuv420p' # segun esto es por compatibilidad
        )
        
        # (overwrite_output=True equivale a -y)
        ffmpeg.run(output_stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        print(f"✓ Video procesado con éxito: {output_path}")
        
    except ffmpeg.Error as e:
        print(f"Error en FFmpeg: {e.stderr.decode('utf-8')}")

# Ejemplo de uso:
if __name__ == "__main__":
    input_vid  = "pruebaGuanajuato7.mp4"
    output_vid = "pruebaGuanajuato7.mp4"
    
    input_vid = "PruebasPersonalesRaw/" + input_vid
    output_vid = "PruebasPersonales/" + output_vid


    # formatear los resultados por compatibilidad
    corto = "pruebaGuanajuato2"
    input_vid = "PruebasPersonalesResult/" + "pruebaGuanajuato2_result.mp4"
    output_vid = "PruebasPersonalesResult/" + "pruebaGuanajuato2_result.mp4"
    
    
    # ejecutar
    preprocess_video(input_vid, output_vid, start_time=29, duration=8)