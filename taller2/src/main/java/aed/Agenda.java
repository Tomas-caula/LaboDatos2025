package aed;

public class Agenda {

    public Fecha fechaActual;
    public Recordatorio[] recordatorios = new Recordatorio[0];

    public Agenda(Fecha fechaActual) {
        // Implementar
        this.fechaActual = fechaActual;
    }

    public void agregarRecordatorio(Recordatorio recordatorio) {
        // Implementar
        Recordatorio[] nuevosRecordatorios = new Recordatorio[recordatorios.length + 1];
        for (int i = 0; i < recordatorios.length; i ++) {
            nuevosRecordatorios[i] = recordatorios[i];
        }
        nuevosRecordatorios[nuevosRecordatorios.length -1] = recordatorio;
        recordatorios = nuevosRecordatorios;
    }

    @Override
    public String toString() {
        // Implementar
        String res = fechaActual.toString() + "\n=====\n";
        for (int i = 0; i < recordatorios.length; i ++){
            if(recordatorios[i].fecha().equals(fechaActual)){
            res +=  recordatorios[i].toString() + "\n";
            }
        }
        return res;
    }

    public void incrementarDia() {
        this.fechaActual.incrementarDia();
        Recordatorio[] nuevRecordatorios = new Recordatorio[0];
        for (int i = 0; i < recordatorios.length; i ++){
            if (recordatorios[i].fecha().mes > this.fechaActual.mes || (recordatorios[i].fecha().mes <= this.fechaActual.mes && recordatorios[i].fecha().dia <= this.fechaActual.dia) ){
                Recordatorio[] copia = new Recordatorio[nuevRecordatorios.length + 1];
                for (int j = 0; j < nuevRecordatorios.length; j++){
                    copia[j] = nuevRecordatorios[j];
                }
                copia[copia.length - 1] = recordatorios[i];
                nuevRecordatorios = copia;  
            }
        }
    
    }

    public Fecha fechaActual() {
        // Implementar
        return this.fechaActual;
    }

}
