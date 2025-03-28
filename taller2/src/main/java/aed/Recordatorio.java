package aed;

public class Recordatorio {

    private Fecha fecha;
    public Horario horario;
    public String mensaje = "";


    public Recordatorio(String mensaje, Fecha fecha, Horario horario) {
    
        this.fecha = new Fecha(fecha);
        this.mensaje = mensaje;
        this.horario = horario;
    }

    public Horario horario() {
        // Implementar
        return this.horario;
    }

    public Fecha fecha() {
        // Implementar
        return new Fecha(this.fecha);
    }

    public String mensaje() {
        // Implementar
        return this.mensaje;
    }

    @Override
    public String toString() {
        // Implementar
        return this.mensaje + " @ " + this.fecha.toString()+ " " + this.horario.toString();
    }

    @Override
    public boolean equals(Object otro) {
        if (otro != null && otro.getClass() == this.getClass()){
            Recordatorio cp = (Recordatorio) otro;
            if(cp.mensaje == this.mensaje && cp.fecha == this.fecha && cp.horario == this.horario){
                return true;
            }
        }
        return false;
    }

}
